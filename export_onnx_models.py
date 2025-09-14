from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.onnx import FeaturesManager, export

MODEL_ID = "Helsinki-NLP/opus-mt-es-en"
OUT_DIR = Path("./models/opus_mt_es_en_onnx")
OUT_DIR.mkdir(parents=True, exist_ok=True)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

onnx_cfg_cls = FeaturesManager.get_config(
    model_type=model.config.model_type,   # "marian"
    feature="seq2seq-lm",
)
onnx_cfg = onnx_cfg_cls(model.config)

export(
    preprocessor=tokenizer,
    model=model,
    config=onnx_cfg,
    opset=17,                                  # >=14 to support SDPA
    output=OUT_DIR / "model.onnx",             # <-- file, not dir
)
print("Exported:", (OUT_DIR / "model.onnx").resolve())
