"""Area D - Efficient models: post-training dynamic INT8 quantization of the fine-tuned BERT.

All nn.Linear layers (attention Q/K/V/output, feed-forward, classifier) get INT8 weights;
activations are quantized on the fly at inference. Runs on CPU (qnnpack on Apple Silicon /
ARM, fbgemm on x86)."""
import platform
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

MODELS = Path(__file__).resolve().parent.parent / "models"
FP32_DIR, INT8_DIR = MODELS / "bert_fp32", MODELS / "bert_int8"


def set_engine():
    engines = torch.backends.quantized.supported_engines
    is_arm = platform.machine().lower() in ("arm64", "aarch64")
    for e in (("qnnpack", "fbgemm", "x86") if is_arm else ("x86", "fbgemm", "qnnpack")):
        if e in engines:
            torch.backends.quantized.engine = e
            return e
    return torch.backends.quantized.engine


def _quantize_dynamic(model):
    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError as e:  # torch.ao.quantization was removed in this torch build
        raise SystemExit("This torch version has no torch.ao.quantization. "
                         "Install the version in requirements.txt.") from e
    return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def load_int8(int8_dir=INT8_DIR):
    """Rebuild the architecture, apply the same quantization, load the saved INT8 weights."""
    set_engine()
    cfg = AutoConfig.from_pretrained(int8_dir)
    model = _quantize_dynamic(AutoModelForSequenceClassification.from_config(cfg).eval())
    model.load_state_dict(torch.load(int8_dir / "model_int8.pt", map_location="cpu", weights_only=False))
    return model.eval()


def main():
    eng = set_engine()
    model = AutoModelForSequenceClassification.from_pretrained(FP32_DIR).eval()
    qmodel = _quantize_dynamic(model)
    INT8_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(qmodel.state_dict(), INT8_DIR / "model_int8.pt")
    for f in FP32_DIR.iterdir():               # config + tokenizer files (no weights)
        if f.suffix not in (".safetensors", ".bin"):
            shutil.copy(f, INT8_DIR / f.name)
    fp = sum(f.stat().st_size for f in FP32_DIR.glob("*.safetensors")) / 2**20
    q = (INT8_DIR / "model_int8.pt").stat().st_size / 2**20
    print(f"[quantize] engine={eng}  fp32={fp:.1f} MB  ->  int8={q:.1f} MB")


if __name__ == "__main__":
    main()
