Hardware: Apple M2, CPU only, 4 threads, Darwin 27.2.0, torch 2.14.1  
Test set: 860 tweets (same for all models). Latency = mean over 200 single-message runs.

| Model | Precision | Macro-F1 | Accuracy | Size (MB) | Latency mean (ms) | p95 (ms) |
|---|---|---|---|---|---|---|
| Rule: edit-distance lexicon | rules | 0.710 | 0.794 | 0.0 | 0.04 | 0.07 |
| tf-idf + LogReg | FP64 | 0.732 | 0.779 | 1.0 | 0.21 | 0.26 |
| word2vec + FFN | FP32 | 0.667 | 0.727 | 3.5 | 0.08 | 0.11 |
| BERT fine-tuned (FP32) | FP32 | 0.802 | 0.852 | 417.7 | 32.70 | 35.31 |
| BERT quantized (INT8) | INT8 dynamic | 0.778 | 0.845 | 173.1 | 33.63 | 50.21 |

FP32 vs INT8 BERT prediction agreement on test: 95.6%
