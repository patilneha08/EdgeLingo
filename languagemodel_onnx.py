import os, numpy as np, onnxruntime as ort
from transformers import AutoTokenizer, AutoConfig

class OnnxTranslator:
    def __init__(self, onnx_model_path: str, tokenizer_name: str, providers=("CPUExecutionProvider",)):
        assert os.path.exists(onnx_model_path), f"ONNX not found: {onnx_model_path}"
        self.session = ort.InferenceSession(onnx_model_path, providers=list(providers))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.cfg = AutoConfig.from_pretrained(tokenizer_name)

        # Marian convention: decoder_start_token_id defaults to pad_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.decoder_start_id = getattr(self.cfg, "decoder_start_token_id", self.pad_id)

        # quick sanity check
        need = {"input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask"}
        have = {i.name for i in self.session.get_inputs()}
        assert need.issubset(have), f"Model inputs mismatch. Need {need}, have {have}"

    def translate(self, text: str, max_new_tokens=128):
        enc = self.tokenizer(text, return_tensors="np", padding=True, truncation=True)
        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)

        # Seed the decoder with a single start token
        decoder_input_ids = np.array([[self.decoder_start_id]], dtype=np.int64)
        decoder_attention_mask = np.ones_like(decoder_input_ids, dtype=np.int64)

        # Greedy autoregressive loop
        for _ in range(max_new_tokens):
            outputs = self.session.run(
                ["logits"],  # first output is logits
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "decoder_input_ids": decoder_input_ids,
                    "decoder_attention_mask": decoder_attention_mask,
                },
            )
            logits = outputs[0]                            # (batch, dec_len, vocab)
            next_id = int(np.argmax(logits[0, -1, :]))     # greedy
            # stop on EOS
            if next_id == self.eos_id:
                break
            # append token
            decoder_input_ids = np.concatenate(
                [decoder_input_ids, np.array([[next_id]], dtype=np.int64)], axis=1
            )
            decoder_attention_mask = np.ones_like(decoder_input_ids, dtype=np.int64)

        # Decode the generated tokens (skip the initial start token)
        gen_ids = decoder_input_ids[0, 1:].tolist()
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# if __name__ == "__main__":
#     # Spanish → English
#     es_en_path = "models/opus_mt_es_en_onnx/model.onnx"
#     es_en_tok = "Helsinki-NLP/opus-mt-es-en"
#     es_en_translator = OnnxTranslator(es_en_path, es_en_tok, providers=("CPUExecutionProvider",))
#     print("ES→EN:", es_en_translator.translate("¿Cómo estás? Esto es una prueba."))

#     # English → Spanish
#     en_es_path = "models/opus_mt_en_es_onnx/model.onnx"
#     en_es_tok = "Helsinki-NLP/opus-mt-en-es"
#     en_es_translator = OnnxTranslator(en_es_path, en_es_tok, providers=("CPUExecutionProvider",))
#     print("EN→ES:", en_es_translator.translate("How are you? This is a test."))
