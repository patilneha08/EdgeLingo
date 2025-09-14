# EdgeLingo
EdgeLingo is an offline, real-time voice translator that runs fully on-device using Snapdragon® X Elite NPU acceleration. It captures speech, transcribes it locally, translates it into the target language, and displays instant captions—delivering low-latency, privacy-preserving communication without internet access

## 📖 Application Description
**EdgeLingo** is an **edge AI application** that runs fully offline on a Snapdragon-powered Windows ARM64 laptop.  
It performs the following pipeline locally, leveraging the NPU for acceleration:

1. **Automatic Speech Recognition (ASR):**  
   Converts spoken audio into text using **Whisper ONNX**, with the **encoder running on the Qualcomm NPU (QNN Execution Provider)**.

2. **Translation:**  
   Translates the recognized text into a target language (using local translation models such as Argos Translate or ONNX-based MT).

3. **Text-to-Speech (TTS):**  
   Converts the translated text into natural speech for audio playback.

This ensures **low latency, privacy (no cloud calls), and true offline capability**, making it ideal for real-time multilingual communication.

---

## 👥 Team Members
- **Ruchi Jha** –  rj2807@nyu.edu
- **Satvik Upadhyay** –  su2250@nyu.edu
- **Neha Patil** –  np2998@nyu.edu


---

## ⚙️ Setup Instructions (from scratch)

### 1. System Requirements
- Windows 11 on ARM64 (Snapdragon-powered laptop with NPU)
- Python 3.12 (ARM64 build)

### 2. Install Qualcomm AI Engine Direct / QNN SDK
1. Download the **Qualcomm AI Stack – QNN SDK (Windows ARM64)**.  
2. After installation, locate the folder containing `QnnHtp.dll` (e.g.  
   `C:\Qualcomm\AIStack\QNN\<ver>\aarch64-windows-msvc\bin\`).
3. Add the `bin` directory to your system PATH:

```powershell
$env:QNN_HOME = "C:\Qualcomm\AIStack\QNN\<ver>\aarch64-windows-msvc"
$env:PATH = "$env:QNN_HOME\bin;$env:PATH"

3. Create Python Virtual Environment
python -m venv edgelingo
.\edgelingo\Scripts\activate

4. Install Dependencies
pip install -U onnxruntime numpy soundfile librosa tiktoken openai-whisper argostranslate
▶️ Run and Usage Instructions
1. Run Encoder Sanity Check
Verify the NPU setup:
python runners/check_ort.py
You should see QNNExecutionProvider listed.
2. Transcribe Speech to Text
python runners/whisper_onnx_qnn_asr.py `
  --audio samples/hello.wav `
  --encoder models/WhisperEncoder.onnx `
  --decoder models/WhisperDecoder.onnx `
  --language en `
  --task transcribe `
  --perf_mode balanced
This will print the transcript of hello.wav.
3. Translate Text
python runners/translate.py --text "Hello, how are you?" --to es
Example output:
Hola, ¿cómo estás?
4. Text-to-Speech (play translated text)
python runners/tts.py --text "Hola, ¿cómo estás?" --out samples/output.wav
This generates a spoken .wav file.
5. End-to-End Demo
Run the entire pipeline in one command:
python runners/main.py --audio samples/hello.wav --target es
Console output:
[ASR Transcript] Hello, how are you?
[Translation] Hola, ¿cómo estás?
[TTS] Audio saved to samples/output.wav
Audio will also play back automatically.

📊 Performance
First run (QNN EP): slower due to graph compilation (~hundreds of ms).
Subsequent runs: significantly faster due to context caching.
Latency improvement vs CPU-only: 2–5× depending on model size.

📂 Repo Structure
EdgeLingo/
├── models/
│   ├── WhisperEncoder.onnx
│   ├── WhisperDecoder.onnx
├── samples/
│   ├── hello.wav
│   ├── output.wav
├── runners/
│   ├── check_ort.py
│   ├── whisper_onnx_qnn_asr.py
│   ├── translate.py
│   ├── tts.py
│   ├── main.py
📝 Notes
The encoder runs on the Snapdragon NPU (via QNN EP).
The decoder, translation, and TTS can run on CPU, or be moved to NPU if time and operator support allow.
The application works offline — no internet is required.
