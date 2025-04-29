# End-to-End English–Turkish Transformer

A hand-crafted PyTorch pipeline—from raw parallel corpora through BPE tokenization, dynamic curriculum training, custom losses, and multi-strategy inference—to build a state-of-the-art English→Turkish translation model.

> **Status (2025-04-29)**: 
> - Model training is currently in progress
> - Upon completion, the trained model will be uploaded to HuggingFace Hub
> - All data artifacts (raw files, tokenized CSVs, train/val splits) will be released via public Google Drive folder

## 📂 Data Preparation

Aggregated a large English–Turkish parallel corpus, ready for BPE vocabulary training and PyTorch ingestion. All raw files, tokenized CSVs, and train/val splits will be uploaded soon to a public Google Drive folder.

### Datasets & Citations

- SNLI-TR (`nli_tr/snli_tr`)
  - SNLI (570K human-written NLI pairs)
- MultiNLI-TR (matched & mismatched dev + train JSON)
  - MultiNLI (433K multi-genre NLI)
- canbingol/translate_dataset (misc. crowdsourced en–tr pairs)
- Tatoeba (OPUS parallel sentences)
- OPUS Wikipedia v1.0 & MultiHPLT v2 dumps
- Custom hand-translated CSV (with auto-capitalized variants)

### Pipeline Overview

#### NLI Readers
- **NLITRReader**: SNLI-TR ↔ SNLI via Hugging Face `load_dataset`
- **NLITRReader2**: MultiNLI-TR JSON ↔ MultiNLI via pairID join

#### Data Processing Steps
1. **Tatoeba extraction**
   - Parse sentences.tar.bz2 + links.tar.bz2 → filter by language & word count

2. **OPUS downloads**
   - Cached ZIPs for Wikipedia & MultiHPLT; skip if already present

3. **Hand translations**
   - Read hand_translated.csv; append original + capitalized variants

4. **Corpus files**
   - `english_corpus.txt` ← all English sentences
   - `turkish_corpus.txt` ← all Turkish sentences

#### SentencePiece BPE Training
(50,000 tokens, 99.99% character coverage)

```bash
spm_train \
  --input=english_corpus.txt \
  --model_prefix=en_spm \
  --vocab_size=50000 \
  --character_coverage=0.9999 \
  --model_type=bpe \
  --pad_id=0 \
  --unk_id=1 \
  --bos_id=2 \
  --eos_id=3

spm_train \
  --input=turkish_corpus.txt \
  --model_prefix=tr_spm \
  --vocab_size=50000 \
  --character_coverage=0.9999 \
  --model_type=bpe \
  --pad_id=0 \
  --unk_id=1 \
  --bos_id=2 \
  --eos_id=3
```

#### Data Processing Pipeline
1. **Tokenized CSV**
   - `save_to_readable_csv("input_ids.pkl", "label_ids.pkl", "tokenized_data.csv")`

2. **Train/Validation split (98%/2%)**
   - `tokenized_data_train.csv` & `tokenized_data_val.csv` via `train_test_split`

## 🚀 Model Training

### Features

#### Training Framework
- **Dynamic curriculum**: Incrementally increase max_tokens each epoch for faster convergence on short sequences
- **Length-balanced sampling**: Per-length cap of (batches_per_size × batch_size) to enforce uniform length distribution without running into OOM errors
- **BucketBatchSampler**: Sorts by length, shuffles within large buckets, yields mini-batches of similar lengths → minimal padding, uniform GPU usage
- **Teacher-forcing hack**: Mask out EOS in decoder input so model cannot "peek" at EOS token

#### Advanced Training Features
- **Auxiliary EOS loss**: Gaussian ramp + peak + tail target distribution on EOS positions; KL-divergence on predicted EOS probabilities encourages correct termination
- **Mixed-precision support** with silent fallback to full precision to avoid NaNs
- **Gradient clipping** (0.5), AdamW, custom LR scheduler: warm-up → plateau → cosine annealing decay
- **Periodic BLEU validation** (2.5% of validation batches) using SacreBLEU
- **Discord & Telegram notifications** of progress, epoch summaries and rich embeds

### Architecture
Hand-implemented Transformer (Vaswani et al.) with:
- 5 encoder & 5 decoder layers
- Embedding dim = 512, FFN = 2048, heads = 8, dropout = 0.2
- Pre-norm attention + SiLU activations
- All model parameters/hyperparameters are customizable

## 🔍 Inference API

### InferBottle Class
Supports multiple decoding strategies out-of-the-box:
- **Greedy** (with repetition penalty)
- **Top-p sampling** (p=0.9, temperature=0.8, repetition penalty)
- **Beam search** (width = 5, log-prob scoring, repetition suppression)
- **"all" mode** returns greedy, top-p, and beam outputs

### Usage Example
```python
from infer import InferBottle

infer = InferBottle(
  model_path="model/best_model.pth",
  en_tokenizer_path="data/en_spm.model",
  tr_tokenizer_path="data/tr_spm.model",
  sampling_method="all",
  beam_width=5
)
out = infer.translate("I'm tired, but I can't sleep.")
print(out["beam_search"])
```
or directly use InferBottle.py

## 🎯 Highlights & Impact

- **Entirely Hand-crafted Pipeline**: Raw archives → translation outputs, no black-box libraries
- **Advanced Sampling & Losses**: EOS KL-loss, curriculum length, bucketized batches
- **Real-time Remote Logging**: Monitor multi-hour training on phone
- **Modular & Extensible**: Swap vocab size, model depth, scheduler, decoding
