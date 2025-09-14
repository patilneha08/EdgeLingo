# languagemodel.py
from languagemodel_onnx import OnnxTranslator   # import the class from the other file

# create singletons so models load only once
_ES2EN = OnnxTranslator(
    "models/opus_mt_es_en_onnx/model.onnx", "Helsinki-NLP/opus-mt-es-en"
)
_EN2ES = OnnxTranslator(
    "models/opus_mt_en_es_onnx/model.onnx", "Helsinki-NLP/opus-mt-en-es"
)

def run_translation(text: str, targetLanguage: str) -> str:
    print("run_translation", targetLanguage)
    if targetLanguage == "E":
        return _ES2EN.translate(text)
    else:
        return _EN2ES.translate(text)

if __name__ == "__main__":
    print(run_translation("¿Cómo estás? Esto es una prueba.", "E"))
    print(run_translation("How are you? This is a test.", "S"))
