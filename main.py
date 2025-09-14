import dearpygui.dearpygui as dpg
import threading
import time
from pathlib import Path
from languagemodel import run_translation
from text_to_speech_model import run_tts
from playsound import playsound

audio_file_path = None  # Global to store wav path

def run_process():
    global audio_file_path
    dpg.configure_item("process_btn", enabled=False)
    dpg.set_value("output_text", "Processing... Please wait.")
    dpg.configure_item("play_audio_btn", show=False, enabled=False)

    def worker():
        global audio_file_path
        input_text = dpg.get_value("input_text").strip()
        if not input_text:
            dpg.set_value("output_text", "No input text provided.")
            dpg.configure_item("process_btn", enabled=True)
            return

        target_lang = "E" if dpg.get_value("lang_radio") == "English" else "S"
        do_translate = dpg.get_value("translate_radio") == "Yes"
        do_audio = dpg.get_value("audio_radio") == "Yes"

        if do_translate:
            try:
                translated_text = run_translation(input_text, target_lang)
            except Exception as e:
                dpg.set_value("output_text", f"Translation error: {e}")
                dpg.configure_item("process_btn", enabled=True)
                return
        else:
            translated_text = input_text

        dpg.set_value("output_text", translated_text)

        audio_file_path = None

        if do_audio:
            try:
                wav_path = call_tts_and_locate_wav(translated_text, target_lang)
                if wav_path:
                    audio_file_path = wav_path
                    dpg.configure_item("play_audio_btn", show=True, enabled=True)
                    dpg.set_value("output_text", translated_text + "\n\nAudio file generated. Click 'Play Audio' to listen.")
                else:
                    dpg.set_value("output_text", translated_text + "\n\nAudio file not found after TTS.")
                    dpg.configure_item("play_audio_btn", show=False, enabled=False)
            except Exception as e:
                dpg.set_value("output_text", translated_text + f"\n\nTTS error: {e}")
                dpg.configure_item("play_audio_btn", show=False, enabled=False)
        else:
            dpg.configure_item("play_audio_btn", show=False, enabled=False)

        dpg.configure_item("process_btn", enabled=True)

    threading.Thread(target=worker, daemon=True).start()

def play_audio():
    global audio_file_path
    if audio_file_path and audio_file_path.exists():
        playsound(str(audio_file_path))

def call_tts_and_locate_wav(text: str, target_lang: str) -> Path | None:
    before = _snapshot_wavs()
    t0 = time.time()
    ret = run_tts(text, target_lang)

    if isinstance(ret, (str, Path)):
        p = Path(ret)
        if p.exists():
            return p

    time.sleep(0.1)
    after = _snapshot_wavs()
    new_paths = [p for p in after if p not in before]
    if new_paths:
        newest = max(new_paths, key=lambda p: p.stat().st_mtime)
        if newest.stat().st_mtime >= t0:
            return newest

    if after:
        newest_any = max(after, key=lambda p: p.stat().st_mtime)
        if newest_any.stat().st_mtime >= t0:
            return newest_any
    return None

def _snapshot_wavs():
    cwd = Path.cwd()
    paths = list(cwd.glob("*.wav"))
    for sub in [p for p in cwd.iterdir() if p.is_dir()]:
        paths.extend(sub.glob("*.wav"))
    return paths

def main():
    dpg.create_context()

    with dpg.window(label="Translator + TTS (DearPyGui)", width=600, height=500):
        dpg.add_text("Input Text:")
        dpg.add_input_text(tag="input_text", multiline=True, height=100, width=-1)

        dpg.add_separator()

        dpg.add_text("Target Language:")
        dpg.add_radio_button(("English", "Spanish"), tag="lang_radio", horizontal=True, default_value="English")

        dpg.add_separator()

        dpg.add_text("Do you want to translate this text?")
        dpg.add_radio_button(("Yes", "No"), tag="translate_radio", horizontal=True, default_value="No")

        dpg.add_separator()

        dpg.add_text("Do you want an audio file?")
        dpg.add_radio_button(("Yes", "No"), tag="audio_radio", horizontal=True, default_value="Yes")

        dpg.add_separator()

        dpg.add_button(label="Process", tag="process_btn", callback=run_process)

        dpg.add_button(label="Play Audio", tag="play_audio_btn", callback=play_audio, show=False, enabled=False)

        dpg.add_separator()

        dpg.add_text("Output Text / Status:")
        dpg.add_input_text(tag="output_text", multiline=True, height=150, width=-1, readonly=True)

    dpg.create_viewport(title='EdgeLingo (DearPyGui)', width=600, height=600)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()
