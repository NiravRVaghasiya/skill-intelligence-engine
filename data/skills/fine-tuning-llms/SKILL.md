---
name: fine-tuning-llms
display_name: Fine-Tuning LLMs
description: >
  Use when the user wants to adapt a pretrained LLM's weights to a specific task
  or domain — preparing an instruction dataset and training a LoRA/QLoRA adapter
  with PEFT rather than prompting or retrieving. Trigger phrases: "fine-tune this
  model", "train a LoRA adapter", "QLoRA setup", "prep a dataset for
  instruction tuning", "why is my fine-tuned model overfitting". NOT for
  steering behavior via the prompt alone (see prompt-engineering), grounding
  answers in a document corpus at inference time (see rag-pipeline), or the
  transformer/attention internals being adapted (see attention-mechanisms).
type: workflow
domain: llm
level: advanced
lifecycle: stable
risk_level: medium
evidence_level: established-practice
last_verified: 2026-09-21
capabilities:
  - lora-adapter-training
  - qlora-quantization
  - instruction-dataset-prep
  - peft-configuration
  - adapter-merging
requires:
conflicts:
related:
  - attention-mechanisms
  - prompt-engineering
  - rag-pipeline
  - llm-evaluation
  - training-deep-models
inputs: An instruction dataset (prompt/response pairs) and a pretrained base model to adapt.
outputs: A saved LoRA adapter (and optionally a merged standalone model) trained toward the target task/domain/format.
version_constraints:
  - "peft: LoraConfig / get_peft_model / prepare_model_for_kbit_training API stable since the 0.4+ line (model-knowledge estimate, not live-verified this session)"
---

## Overview
Fine-tuning updates an LLM's weights (or a small set of added weights) so behavior
that a prompt alone can't reliably produce — a house style, a domain vocabulary,
a strict output format — becomes baked in. This skill covers the practical,
parameter-efficient path: dataset prep, QLoRA (4-bit base model + LoRA adapters)
via Hugging Face `transformers`/`peft`/`bitsandbytes`, and saving/merging the
result. It assumes you already have a reason to fine-tune rather than prompt or
retrieve — if you don't, `prompt-engineering` or `rag-pipeline` are cheaper first.

## Workflow
1. **Prepare the instruction dataset.** Format examples as consistent
   prompt/response pairs and tokenize with the same template the base model was
   instruction-tuned with (or a fixed custom template if training from a plain
   base model).
   ```python
   from datasets import load_dataset

   raw = load_dataset("json", data_files="train.jsonl")["train"]
   # Each record: {"instruction": ..., "input": ..., "output": ...}

   def to_text(example):
       prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
       return {"text": prompt + example["output"]}

   dataset = raw.map(to_text)
   ```
2. **Load the base model in 4-bit (QLoRA) and its tokenizer.** Quantizing the
   frozen base weights to 4-bit slashes memory so larger models fit on a single
   GPU; only the LoRA adapters train in higher precision.
   ```python
   import torch
   from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

   model_id = "meta-llama/Llama-3.1-8B"
   bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_compute_dtype=torch.bfloat16,
       bnb_4bit_use_double_quant=True,
   )
   tokenizer = AutoTokenizer.from_pretrained(model_id)
   tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

   model = AutoModelForCausalLM.from_pretrained(
       model_id, quantization_config=bnb_config, device_map="auto"
   )
   ```
3. **Wrap the model with a LoRA adapter via PEFT.** Prepare the quantized model
   for k-bit training first, then attach low-rank adapters to the attention
   projection layers.
   ```python
   from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

   model = prepare_model_for_kbit_training(model)

   lora_config = LoraConfig(
       r=16, lora_alpha=32, lora_dropout=0.05,
       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # model-specific
       task_type="CAUSAL_LM",
   )
   model = get_peft_model(model, lora_config)
   model.print_trainable_parameters()  # sanity-check: should be << total params
   ```
4. **Train with `SFTTrainer` (TRL).** Handles tokenization, packing, and the
   training loop for supervised fine-tuning on the `text` field.
   ```python
   from trl import SFTTrainer, SFTConfig

   sft_config = SFTConfig(
       output_dir="./lora-out",
       per_device_train_batch_size=4,
       gradient_accumulation_steps=4,
       num_train_epochs=3,
       learning_rate=2e-4,          # LoRA tolerates/needs a higher LR than full FT
       bf16=True,
       logging_steps=10,
       save_strategy="epoch",
       max_seq_length=1024,
   )
   trainer = SFTTrainer(
       model=model,
       train_dataset=dataset,
       args=sft_config,
   )
   trainer.train()
   ```
5. **Save the adapter, and optionally merge it into the base model.** Saving the
   adapter alone keeps it small (MBs) and swappable; merging produces a single
   deployable model at the cost of losing that flexibility.
   ```python
   model.save_pretrained("./lora-out/adapter")   # small, swappable adapter only

   # Optional: merge for a standalone deployable model
   from peft import PeftModel
   merged = PeftModel.from_pretrained(model, "./lora-out/adapter").merge_and_unload()
   merged.save_pretrained("./lora-out/merged-model")
   ```

## Gotchas
- **`target_modules` must match the actual layer names for the model family.**
  `q_proj`/`v_proj` works for Llama-style models but not every architecture —
  inspect `model.named_modules()` first if adapters silently attach to nothing.
- **Skipping `prepare_model_for_kbit_training`** before wrapping with LoRA breaks
  gradient flow through the quantized base (frozen layers need casting/gradient
  checkpointing set up correctly) — training will run but barely learn.
- **LoRA needs a higher learning rate than full fine-tuning** (typically 1e-4 to
  2e-4 vs. 1e-5 to 5e-5) because only a small fraction of parameters are updated;
  reusing a full-FT LR under-trains the adapter.
- **Overfitting on small instruction sets.** LoRA's low parameter count reduces
  but does not eliminate overfitting risk on datasets under a few thousand
  examples — hold out an eval split and watch validation loss, don't just trust
  training loss going down.
- **Tokenizer padding side for causal LMs.** Training typically pads right, but
  batched generation afterward needs left-padding — using the wrong side for the
  wrong phase produces garbled outputs or silently wrong loss masking.
- **Forgetting to freeze/quantize correctly leads to OOM.** 4-bit quantization
  only reduces the *base* model's memory; gradient accumulation, optimizer states
  for the adapter, and activation memory (mitigated by gradient checkpointing)
  still need to fit — profile before scaling batch size up.

## References
- [Hu et al., 2021 — LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — the adapter method used throughout this workflow.
- [Dettmers et al., 2023 — QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — the 4-bit quantization + LoRA combination in steps 2–3.
- [Hugging Face PEFT documentation](https://huggingface.co/docs/peft/index) — `LoraConfig`, `get_peft_model`, and merging APIs used above.
