from languagemodel_onnx import run_translation

text_es = "Hola, ¿cómo estás?"
result_en = run_translation(text_es, "E")
print("Spanish to English:", result_en)

text_en = "Hello, how are you?"
result_es = run_translation(text_en, "S")  # or whatever your else condition is
print("English to Spanish:", result_es)
