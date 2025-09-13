# -*- coding: utf-8 -*-

from transformers import pipeline

# English -> Spanish
en2es = pipeline(
    task="translation",
    model="Helsinki-NLP/opus-mt-en-es",
    device_map="auto"
)

# Spanish -> English
es2en = pipeline(
    task="translation",
    model="Helsinki-NLP/opus-mt-es-en",
    device_map="auto"
)

en_text = "How are you? I'm looking for a good restaurant near the museum."
es_out = en2es(en_text, max_length=256)[0]["translation_text"]
print("EN → ES:", es_out)

es_text = "¿Cómo estás? Estoy buscando un buen restaurante cerca del museo."
en_out = es2en(es_text, max_length=256)[0]["translation_text"]
print("ES → EN:", en_out)

en_text = "The head of the United Nations says there is no military solution in Syria"
es_out = en2es(en_text, max_length=256)[0]["translation_text"]
print("EN → ES:", es_out)

es_text = "El jefe de la ONU afirmó que no existe una solución militar en Siria"
en_out = es2en(es_text, max_length=256)[0]["translation_text"]
print("ES → EN:", en_out)

