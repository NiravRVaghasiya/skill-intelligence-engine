---
name: nlp-tasks
display_name: NLP Tasks (Tokenization, NER, Classification, Summarization)
description: >
  Use when the user wants to tokenize text, run named entity recognition,
  classify documents, or summarize text using classical/transformer NLP
  pipelines. Trigger phrases: "tokenize this text", "extract entities from
  this document", "classify these support tickets", "summarize this
  article", "build an NER pipeline", "fine-tune a text classifier". NOT for
  open-ended generation or few-shot prompting of large LLMs (see
  prompt-engineering) or retrieval-augmented question answering (see
  rag-pipeline).
type: workflow
domain: specialized
level: intermediate
lifecycle: stable
risk_level: low
evidence_level: official-documentation
last_verified: 2026-09-21
capabilities:
  - word-subword-tokenization-comparison
  - named-entity-recognition-pipeline
  - transformer-text-classification-finetuning
  - long-document-summarization-chunking
  - nlp-task-metric-selection
requires:
conflicts:
related:
  - attention-mechanisms
  - rnn-sequence
  - prompt-engineering
  - rag-pipeline
inputs: Raw text documents (single strings or a corpus) needing tokenization, entity extraction, classification labels, or summarization.
outputs: A configured tokenizer/pipeline plus task-appropriate evaluation metrics (seqeval F1, macro-F1, ROUGE) for the chosen NLP task.
---

## Overview
Covers the four workhorse NLP tasks — tokenization, named entity recognition
(NER), text classification, and summarization — using spaCy and Hugging Face
Transformers pipelines. The focus is task-specific supervised/pretrained
models (encoder classifiers, seq2seq summarizers), not prompting a generative
LLM. The user walks away with a runnable pipeline for each task plus the
right metric to evaluate it.

## Workflow
1. **Tokenize and inspect.** Word-level (spaCy) and subword (Hugging Face)
   tokenization behave differently — check both before choosing a model,
   since NER label alignment depends on it.
   ```python
   import spacy
   from transformers import AutoTokenizer

   nlp = spacy.load("en_core_web_sm")
   doc = nlp("Amazon's new office opened in Seattle on Jan 5th.")
   print([(t.text, t.pos_) for t in doc])

   tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
   enc = tokenizer("Amazon's new office opened in Seattle.", return_offsets_mapping=True)
   print(tokenizer.convert_ids_to_tokens(enc["input_ids"]))
   ```
2. **Named entity recognition.** Start with a pretrained pipeline; fall back
   to a transformer NER model for finer-grained or noisier text.
   ```python
   from transformers import pipeline

   ner = pipeline(
       "token-classification",
       model="dslim/bert-base-NER",
       aggregation_strategy="simple",
   )
   for ent in ner("Amazon's new office opened in Seattle on Jan 5th."):
       print(ent["word"], ent["entity_group"], round(ent["score"], 3))
   ```
3. **Text classification.** For domain-specific labels, fine-tune a small
   transformer encoder rather than relying on zero-shot.
   ```python
   from datasets import load_dataset
   from transformers import (
       AutoTokenizer, AutoModelForSequenceClassification,
       TrainingArguments, Trainer,
   )

   ds = load_dataset("imdb")
   tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

   def tokenize(batch):
       return tokenizer(batch["text"], truncation=True, max_length=256)

   ds_tok = ds.map(tokenize, batched=True)
   model = AutoModelForSequenceClassification.from_pretrained(
       "distilbert-base-uncased", num_labels=2
   )
   args = TrainingArguments(
       output_dir="clf-out", per_device_train_batch_size=16,
       num_train_epochs=2, evaluation_strategy="epoch",
   )
   trainer = Trainer(
       model=model, args=args,
       train_dataset=ds_tok["train"].shuffle(seed=42).select(range(2000)),
       eval_dataset=ds_tok["test"].select(range(500)),
   )
   trainer.train()
   ```
4. **Summarization.** Use a pretrained seq2seq summarizer; chunk long
   documents to stay under the model's max input length.
   ```python
   from transformers import pipeline

   summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

   def summarize_long(text, chunk_chars=3000):
       chunks = [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)]
       partials = [
           summarizer(c, max_length=120, min_length=30, do_sample=False)[0]["summary_text"]
           for c in chunks
       ]
       return summarizer(" ".join(partials), max_length=150, min_length=40)[0]["summary_text"]

   print(summarize_long(long_article_text))
   ```
5. **Evaluate with task-appropriate metrics.** Accuracy alone is misleading
   for NER/classification with imbalanced labels; ROUGE is standard for
   summarization.
   ```python
   import evaluate

   seqeval = evaluate.load("seqeval")          # NER: precision/recall/F1 per entity type
   rouge = evaluate.load("rouge")               # summarization
   f1 = evaluate.load("f1")                     # classification, use average="macro"

   print(rouge.compute(predictions=[pred_summary], references=[ref_summary]))
   ```

## Gotchas
- **Subword/word label misalignment.** Transformer tokenizers split words
  into subword pieces, so word-level NER labels must be re-aligned using
  `tokenizer(...).word_ids()` — naively assigning one label per token
  corrupts the training signal.
- **Silent truncation on summarization inputs.** BART/T5 summarizers have a
  hard input limit (e.g., 1024 tokens for `facebook/bart-large-cnn`); text
  beyond that is silently dropped unless you chunk and merge as above.
- **Class imbalance skews accuracy.** Support-ticket or spam classifiers are
  rarely balanced — report macro-F1 or per-class precision/recall, not raw
  accuracy, or a majority-class classifier will look "good."
- **Pretrained NER label sets are generic.** `dslim/bert-base-NER` and
  spaCy's default models recognize PERSON/ORG/GPE/MISC only — domain entities
  (product SKUs, drug names) need a fine-tuned or gazetteer-augmented model.
- **Tokenizer/model mismatch.** Loading a tokenizer from a different model
  family than the one you fine-tune (e.g., BERT tokenizer with a RoBERTa
  model) silently produces garbage inputs — always pair `AutoTokenizer` and
  `AutoModel*` from the same checkpoint name.

## References
- [Hugging Face Transformers: Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines) — canonical reference for task pipelines (NER, classification, summarization).
- [spaCy 101](https://spacy.io/usage/spacy-101) — tokenization and linguistic annotation fundamentals.
- [Hugging Face: Token classification guide](https://huggingface.co/docs/transformers/tasks/token_classification) — the correct way to align subword tokens with NER labels.
