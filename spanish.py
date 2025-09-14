from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import soundfile as sf

MODEL_NAME = "openai/whisper-small"  # or tiny/small/medium; choose multilingual
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.eval()

# Force Spanish + transcribe (not translate)
forced_ids = processor.get_decoder_prompt_ids(language="spanish", task="transcribe")

audio, sr = sf.read("spanish.wav")
inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        forced_decoder_ids=forced_ids,   # <-- critical
        max_new_tokens=225
    )

text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(text)  # should be Spanish text