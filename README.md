# Multimodal Depression Severity Prediction from Clinical Interview Audio and Facial Action Units

A comparative study of CNN, CNN-LSTM, CNN-Transformer, Wav2Vec2, and multimodal fusion architectures for predicting PHQ-8 depression severity from the [DAIC-WOZ](https://dcapswoz.ict.usc.edu/) clinical interview corpus.

> **TL;DR:** This project builds a complete, reproducible deep learning pipeline on a restricted clinical dataset, compares 9 architecture configurations, and — using SHAP explainability and modality-ablation testing — discovers and documents that the "best" model's apparent performance is actually mean-collapse (predicting near the population average) rather than genuine learned signal. The diagnostic process is the real contribution here, not a headline accuracy number.

---

## What this is

A single, self-contained Google Colab notebook (`Multimodal_Depression_Prediction.ipynb`) that takes raw DAIC-WOZ interview data all the way through to a ranked, explainable, statistically-tested comparison of depression-severity prediction architectures.

**Pipeline:** data acquisition → EDA → facial Action Unit feature extraction → Wav2Vec2 audio embeddings → sequence construction → 7 base architectures + Optuna-tuned variants → SHAP/attention/embedding explainability → ablation testing → statistical significance testing → auto-generated research report.

## Why this dataset, and an important caveat

DAIC-WOZ is a restricted-access clinical dataset requiring a signed data use agreement with USC. **It also does not ship raw interview video** — for participant de-identification, USC instead distributes pre-extracted OpenFace facial behavior features (2D/3D landmarks, gaze, head pose, and 17 Action Units) alongside raw audio. This pipeline is built around that reality: facial behavior is modeled via Action Unit intensities, not learned CNN features from frames.

The notebook includes a synthetic-data fallback mode (`CONFIG.use_synthetic_data = True`) that generates a structurally faithful mock corpus, so the full pipeline is runnable and inspectable even without DAIC-WOZ access. Real-data results below are from the actual 189-session DAIC-WOZ corpus.

## Results

| Rank | Architecture | Backbone | Params | MAE | R² | Pearson r |
|---|---|---|---|---|---|---|
| 1 | C_CNN_Transformer_TUNED | OpenFace AU | 836K | 4.655 | -0.009 | 0.274 |
| 2 | E3_Fusion_Attention_TUNED | AU + Wav2Vec2 | 672K | 4.672 | -0.040 | -0.158 |
| 3 | E1_Fusion_Early | AU + Wav2Vec2 | 3.4M | 4.684 | -0.073 | 0.107 |
| 4 | B_CNN_LSTM | OpenFace AU | 638K | 4.707 | -0.122 | 0.017 |
| 5 | E2_Fusion_Late | AU + Wav2Vec2 | 3.4M | 4.746 | -0.149 | 0.148 |
| 6 | C_CNN_Transformer | OpenFace AU | 1.6M | 4.761 | -0.169 | 0.066 |
| 7 | E3_Fusion_Attention | AU + Wav2Vec2 | 4.0M | 4.789 | -0.208 | NaN |
| 8 | D_AudioOnly_Wav2Vec2 | Wav2Vec2 | 107K | 5.528 | -0.629 | 0.067 |
| 9 | A_FacialCNN | OpenFace AU | 11K | 6.600 | -1.231 | NaN |

189 real DAIC-WOZ sessions · 132 train / 28 val / 29 test (stratified by PHQ-8 severity band) · 9 configurations including Optuna-tuned Transformer and Attention-Fusion variants.

### The honest finding

The table above ranks `C_CNN_Transformer_TUNED` first, with a Pearson correlation of 0.274 that looks like a real, if modest, predictive signal. **It probably isn't.** The notebook's explainability and ablation sections (Section 14) and an automated diagnostic check (Section 6.5 of the generated report) provide converging evidence of mean-collapse instead:

- The best model's predictions have a standard deviation of **0.019** against a true-label standard deviation of **5.89** — it is predicting almost the same value for every session, regardless of input.
- **SHAP attribution is ~0.0000** for both facial and audio dimensions — the model isn't meaningfully using either modality's content.
- **Zeroing out either modality entirely changes test MAE by at most 0.003 points** — removing the actual input data barely affects the output.
- The headline r=0.274 does not clear the statistical significance threshold for a 29-example test set (|r| > 0.367 needed at p<0.05) — consistent with sampling noise, not learned structure.

This is most likely a **data-scarcity problem**: models with up to ~4M trainable parameters were trained on only 132 labeled examples, against a heavily right-skewed PHQ-8 distribution where predicting near the population mean is a very easy local minimum for gradient descent to find. Notably, an earlier iteration of the sequence-construction step used naive point-sampling (keeping <0.2% of raw frames per session) — replacing it with windowed average pooling (verified to preserve 100% of raw frame information) did not resolve the collapse, which is itself evidence against information loss being the dominant cause.

The notebook's auto-generated report documents this diagnosis programmatically, computed from the actual run — not asserted after the fact.

## Architecture overview

| ID | Name | Input | Temporal modeling |
|---|---|---|---|
| A | Facial CNN Baseline | OpenFace AU embeddings | Mean pooling |
| B | CNN + LSTM | OpenFace AU embeddings | Bidirectional LSTM |
| C | CNN + Transformer | OpenFace AU embeddings | Self-attention |
| D | Audio-Only Wav2Vec2 | Wav2Vec2 embeddings | Mean pooling |
| E1 | Fusion (Early) | AU + Wav2Vec2 | Concatenation |
| E2 | Fusion (Late) | AU + Wav2Vec2 | Learned weighted average |
| E3 | Fusion (Attention) | AU + Wav2Vec2 | Cross-modal attention |

C and E3 additionally have Optuna-tuned variants (8 trials each over learning rate, dropout, model width/depth, attention heads, weight decay).

## Setup

1. **Get DAIC-WOZ access** at https://dcapswoz.ict.usc.edu/ (requires an institutional data use agreement).
2. Open the notebook in Google Colab, run **Section 0** to download the corpus into your Google Drive (one-time, ~92GB — a paid Google One tier is recommended over the free 15GB).
3. Run **Section 1**, then **restart the Colab runtime once** (required — the install cell pins NumPy/Pillow versions and forces PyTorch-only mode for `transformers`; skipping the restart can crash the kernel).
4. Run the rest of the notebook top to bottom. `CONFIG.use_synthetic_data = False` by default; the unzip/metadata-loading steps are resumable and only need to fully run once per Drive — subsequent sessions reuse the cached extraction.

No GPU is strictly required, but Wav2Vec2 embedding extraction over 189 real sessions (up to ~17 minutes each) is dramatically faster with one; a free Colab T4 runtime is sufficient.

## Repository / output structure

```
results/
├── architecture_comparison.csv
├── statistical_testing_results.csv
├── metrics_summary.json
├── final_research_report.md       # auto-generated, includes the mean-collapse diagnosis
├── predictions/<architecture>_test_predictions.csv
└── *.png                          # all figures (EDA, training curves, SHAP, ablation, error analysis)
artifacts/
├── config.json
├── best_hyperparameters.json
└── checkpoints/<architecture>_best.pt
```

## Engineering notes

This pipeline went through substantial real debugging against the actual restricted dataset, documented inline in the notebook:

- **DAIC-WOZ ships no raw video** — the facial pipeline was rebuilt from scratch around OpenFace Action Units after discovering this mid-project.
- **Duplicate-column bug**: `train`/`dev` splits and `full_test_split.csv` use different PHQ-8 column names (`PHQ8_Score` vs `PHQ_Score`); naively merging silently discarded every test-split label until fixed with column coalescing.
- **Memory architecture flaw**: an earlier version held every session's full Wav2Vec2 embedding sequence in RAM simultaneously — fine for short synthetic clips, but real sessions up to 17 minutes long crashed the Colab runtime outright. Fixed via disk-cached, lazily-loaded per-batch embeddings.
- **NumPy/PyTorch ABI conflicts and unwanted TensorFlow auto-initialization** — both diagnosed from raw Colab crash logs and fixed with explicit version pinning and `USE_TF=0`.
- **Google Drive FUSE-mount instability** under sustained write load during dataset extraction — fixed with local-disk staging, batched writes, and automatic retry with backoff.

## Limitations

- Single train/val/test split (no cross-validation across DAIC-WOZ's official folds).
- Facial modeling limited to the standard 17-AU set; no access to raw video.
- Wav2Vec2 used frozen (no fine-tuning on interview speech specifically).
- See `final_research_report.md` Section 6.5 for the full mean-collapse diagnosis and its implications for interpreting the results above.

## Future work

- Fine-tune Wav2Vec2 and explore richer facial features end-to-end rather than frozen embeddings.
- Incorporate the interview transcripts DAIC-WOZ provides as a third modality.
- Cross-validate across official AVEC train/dev/test folds for direct comparability with published benchmarks.
- Investigate whether much smaller, more heavily regularized models (or non-deep baselines) generalize better given how few labeled examples are available.

## Acknowledgments

DAIC-WOZ corpus: Gratch et al., USC Institute for Creative Technologies. Built with PyTorch, HuggingFace Transformers, Optuna, SHAP, and scikit-learn.
