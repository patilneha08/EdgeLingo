import argparse
from languagemodel import run_translation
from text_to_speech_model import run_tts

def ask_lang() -> str:
    while True:
        lang = input("Translate to (E for English / S for Spanish): ").strip().upper()
        if lang in {"E", "S"}:
            return lang
        print("Please enter 'E' or 'S'.")

def ask_yes_no(prompt: str) -> bool:
    while True:
        choice = input(f"{prompt} (Y/N): ").strip().upper()
        if choice in {"Y", "N"}:
            return choice == "Y"
        print("Please enter Y or N.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run translation or TTS")
#     parser.add_argument("--mode", choices=["translate", "tts"], required=True,
#                         help="Choose whether to run text translation or text-to-speech")
#     args = parser.parse_args()

#     translated_text = ""

#     if args.mode == "translate":
#         translated_text = run_translation()
#         print(translated_text)
#     elif args.mode == "tts":
#         run_tts()

def main():
    target_lang = ask_lang()
    text = input("Enter your text: ").strip()
    if not text:
        print("No text provided. Exiting.")
        return

    outputText = run_translation(text, target_lang)
    translate_choice = False
    translate_speech = False

    translate_choice = ask_yes_no("Do you want to translate this text?")
    
    translate_speech_choice = ask_yes_no("Do you want an audio file?")

    run_tts(outputText, target_lang)
    

if __name__ == "__main__":
    main()