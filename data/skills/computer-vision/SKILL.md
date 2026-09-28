---
name: computer-vision
display_name: Computer Vision (Detection, Segmentation, Augmentation)
description: >
  Use when the user wants to detect objects, segment images, or set up
  image augmentation for a computer vision training pipeline. Trigger
  phrases: "detect objects in these images", "train an object detector",
  "segment this image", "build an image augmentation pipeline", "fine-tune
  a detection model on my dataset". NOT for convolutional architecture
  theory or how CNNs work internally (see cnn-vision) or generic training
  loop mechanics (see training-deep-models).
type: workflow
domain: specialized
level: intermediate
lifecycle: stable
risk_level: low
evidence_level: official-documentation
last_verified: 2026-09-21
capabilities:
  - bbox-mask-aware-augmentation
  - object-detector-finetuning
  - segmentation-model-finetuning
  - map-iou-evaluation
  - non-max-suppression-tuning
requires:
conflicts:
related:
  - cnn-vision
  - data-preprocessing
  - model-evaluation
  - training-deep-models
inputs: An image dataset with bounding-box or segmentation-mask annotations plus a target class list.
outputs: A fine-tuned torchvision detector or segmentation model, an augmentation pipeline, and mAP/IoU evaluation results.
version_constraints:
  - "torchvision: enum-based `weights=` API (e.g. `weights=\"DEFAULT\"`), replacing the older `pretrained=True` boolean, available since torchvision 0.13 (2022) — model-knowledge estimate, not live-verified this session."
  - "torchvision: `fasterrcnn_resnet50_fpn_v2` / `_v2` detection model variants added in torchvision 0.13 (2022) — model-knowledge estimate, not live-verified this session."
---

## Overview
Covers the applied computer-vision pipeline: augmenting an image dataset
correctly, fine-tuning a pretrained object detector, and fine-tuning a
pretrained segmentation model, using torchvision and Albumentations. The
user walks away with a training-ready dataset/model pair and the right
metrics (mAP, IoU) to judge them — architecture internals live in
`cnn-vision`.

## Workflow
1. **Augment images and labels together.** Bounding boxes and masks must be
   transformed in sync with the image, or augmentation corrupts the labels.
   ```python
   import albumentations as A
   from albumentations.pytorch import ToTensorV2

   train_tfms = A.Compose(
       [
           A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0)),
           A.HorizontalFlip(p=0.5),
           A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
           A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
           ToTensorV2(),
       ],
       bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
   )
   transformed = train_tfms(image=image, bboxes=boxes, labels=labels)
   ```
2. **Fine-tune an object detector.** Start from a COCO-pretrained
   Faster R-CNN and replace only the classification head for your classes.
   ```python
   import torch
   from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
   from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

   num_classes = 3 + 1  # + background
   model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
   in_features = model.roi_heads.box_predictor.cls_score.in_features
   model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)
   ```
3. **Fine-tune a segmentation model.** DeepLabV3 with a ResNet backbone is a
   solid pretrained baseline for semantic segmentation.
   ```python
   from torchvision.models.segmentation import deeplabv3_resnet50

   seg_model = deeplabv3_resnet50(weights="DEFAULT")
   seg_model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
   seg_model.to(device)
   ```
4. **Run the training loop.** Detection models in torchvision return the loss
   dict directly in train mode.
   ```python
   from torch.utils.data import DataLoader

   loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                        collate_fn=lambda b: tuple(zip(*b)))
   optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)

   model.train()
   for images, targets in loader:
       images = [img.to(device) for img in images]
       targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
       loss_dict = model(images, targets)
       loss = sum(loss_dict.values())
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
   ```
5. **Evaluate with detection/segmentation metrics.** Pixel accuracy hides
   class imbalance; mAP and IoU are the standard metrics.
   ```python
   from torchmetrics.detection import MeanAveragePrecision
   from torchmetrics import JaccardIndex

   map_metric = MeanAveragePrecision(iou_type="bbox")
   map_metric.update(preds, targets)          # preds/targets: list[dict] with boxes/labels/scores
   print(map_metric.compute())                # mAP@[.5:.95], mAP@.5, ...

   iou_metric = JaccardIndex(task="multiclass", num_classes=num_classes).to(device)
   print(iou_metric(seg_logits.argmax(1), seg_masks))
   ```
6. **Run inference and apply non-max suppression.** Pretrained detectors
   apply NMS internally, but re-check the threshold when merging multi-scale
   or ensembled predictions.
   ```python
   model.eval()
   with torch.no_grad():
       preds = model([image.to(device)])[0]
   keep = preds["scores"] > 0.5
   boxes, labels, scores = preds["boxes"][keep], preds["labels"][keep], preds["scores"][keep]
   ```

## Gotchas
- **Transforming image without labels.** Plain `torchvision.transforms`
  applied only to the image (not the boxes/masks) silently desyncs labels
  after crops, flips, or resizes — use Albumentations with `bbox_params`/
  `mask` support, or write the box/mask transform by hand.
- **Normalization mismatch.** Pretrained torchvision models expect ImageNet
  mean/std normalization; skipping or mismatching it tanks accuracy without
  raising an error.
- **Pixel accuracy is misleading for segmentation.** When background pixels
  dominate the mask, a model that predicts all-background scores high
  pixel accuracy — use IoU/Dice per class instead.
- **Aspect-ratio distortion.** Naive resize-to-square stretches small/thin
  objects out of proportion; use letterbox padding (resize + pad) to
  preserve aspect ratio, especially for detection.
- **NMS threshold tuning.** Too aggressive an IoU threshold in non-max
  suppression drops genuinely separate, closely-spaced objects; too loose a
  threshold leaves duplicate boxes — tune per dataset, don't trust the
  torchvision default blindly.

## References
- [torchvision: Object Detection reference](https://pytorch.org/vision/stable/models.html#object-detection) — pretrained detection/segmentation model zoo and fine-tuning API.
- [Albumentations documentation](https://albumentations.ai/docs/) — bbox/mask-aware augmentation pipelines.
- [torchmetrics: Detection](https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html) — correct mAP computation for object detection.
