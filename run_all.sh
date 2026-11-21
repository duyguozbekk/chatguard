#!/usr/bin/env bash
# Reproduce every number in the presentation: data -> 4 (+1) models -> quantization -> evaluation.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models results
python -m src.data                 # dataset statistics
python -m src.train_classical      # rule baseline + tf-idf/LogReg
python -m src.train_w2v_ffn        # own word2vec + feedforward net (SGD)
python -m src.train_bert "$@"      # BERT fine-tuning (MPS/GPU if available)
python -m src.quantize             # dynamic INT8 quantization
python -m src.evaluate             # same test set, macro-F1 / size / latency
echo "Done. Start the app with:  python app.py"
