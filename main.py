import sys
import time
from pathlib import Path

from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QRadioButton, QGroupBox, QLabel, QSlider
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

# --- Your modules ---
from languagemodel import run_translation
from text_to_speech_model import run_tts


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Translator + TTS (Qt Frontend)")
        self.resize(760, 580)

        # Player setup
        self.player = QMediaPlayer()
        self.audio_out = QAudioOutput()
        self.player.setAudioOutput(self.audio_out)
        self.audio_out.setVolume(0.8)

        root = QVBoxLayout(self)

        # Input
        self.in_text = QTextEdit()
        self.in_text.setPlaceholderText("Type or paste text…")
        root.addWidget(self._boxed("Input Text", self.in_text))

        # Radio groups
        root.addWidget(self._make_language_group())
        root.addWidget(self._make_translate_group())
        root.addWidget(self._make_audio_group())

        # Process button
        self.btn_process = QPushButton("Process")
        self.btn_process.clicked.connect(self.process)
        root.addWidget(self.btn_process)

        # Output text
        self.out_text = QTextEdit()
        self.out_text.setReadOnly(True)
        root.addWidget(self._boxed("Output Text / Status", self.out_text))

        # Player UI
        root.addWidget(self._make_player_group())

        # Player signal wiring
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)

    # ---- UI helpers ----
    def _boxed(self, title, widget):
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        lay.addWidget(widget)
        return box

    def _make_language_group(self):
        box = QGroupBox("Target Language")
        h = QHBoxLayout(box)
        self.rb_en = QRadioButton("English")
        self.rb_es = QRadioButton("Spanish")
        self.rb_en.setChecked(True)
        h.addWidget(self.rb_en)
        h.addWidget(self.rb_es)
        return box

    def _make_translate_group(self):
        box = QGroupBox("Do you want to translate this text?")
        h = QHBoxLayout(box)
        self.rb_tr_yes = QRadioButton("Yes")
        self.rb_tr_no = QRadioButton("No")
        self.rb_tr_no.setChecked(True)
        h.addWidget(self.rb_tr_yes)
        h.addWidget(self.rb_tr_no)
        return box

    def _make_audio_group(self):
        box = QGroupBox("Do you want an audio file?")
        h = QHBoxLayout(box)
        self.rb_audio_yes = QRadioButton("Yes")
        self.rb_audio_no = QRadioButton("No")
        self.rb_audio_yes.setChecked(True)
        h.addWidget(self.rb_audio_yes)
        h.addWidget(self.rb_audio_no)
        return box

    def _make_player_group(self):
        box = QGroupBox("Audio Player")
        v = QVBoxLayout(box)

        # Buttons
        row = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_pause = QPushButton("Pause")
        self.btn_stop = QPushButton("Stop")
        self.btn_play.clicked.connect(self.player.play)
        self.btn_pause.clicked.connect(self.player.pause)
        self.btn_stop.clicked.connect(self.player.stop)
        row.addWidget(self.btn_play)
        row.addWidget(self.btn_pause)
        row.addWidget(self.btn_stop)
        v.addLayout(row)

        # Slider + time label
        row2 = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(lambda val: self.player.setPosition(int(val)))
        self.lbl_time = QLabel("00:00 / 00:00")
        row2.addWidget(self.slider, 1)
        row2.addWidget(self.lbl_time)
        v.addLayout(row2)

        return box

    # ---- Player updates ----
    def _on_duration_changed(self, dur_ms: int):
        self.slider.setRange(0, int(dur_ms))
        self._update_time_label(self.player.position(), dur_ms)

    def _on_position_changed(self, pos_ms: int):
        if not self.slider.isSliderDown():
            self.slider.setValue(int(pos_ms))
        self._update_time_label(pos_ms, self.player.duration())

    def _update_time_label(self, pos_ms: int, dur_ms: int):
        def fmt(ms):
            s = int(ms // 1000)
            return f"{s // 60:02d}:{s % 60:02d}"
        self.lbl_time.setText(f"{fmt(pos_ms)} / {fmt(dur_ms)}")

    # ---- Core action ----
    def process(self):
        raw = self.in_text.toPlainText().strip()
        if not raw:
            self.out_text.setPlainText(" No input text provided.")
            return

        target_lang = "E" if self.rb_en.isChecked() else "S"

        # Translate if chosen
        if self.rb_tr_yes.isChecked():
            try:
                processed = run_translation(raw, target_lang)
            except Exception as e:
                self.out_text.setPlainText(f" Translation error: {e}")
                return
        else:
            processed = raw

        # Show output text
        self.out_text.setPlainText(processed)

        # Audio if chosen
        if self.rb_audio_yes.isChecked():
            wav_path = self._call_tts_and_locate_wav(processed, target_lang)
            if wav_path:
                self.player.setSource(QUrl.fromLocalFile(str(wav_path.resolve())))
                self.player.stop()
            else:
                self.out_text.append("\n Audio not found after TTS.")
        else:
            self.player.setSource(QUrl())
            self.player.stop()
            self.out_text.append("\n Skipped audio generation.")

    # Detect WAV file after run_tts
    def _call_tts_and_locate_wav(self, text: str, target_lang: str) -> Path | None:
        before = self._snapshot_wavs()
        t0 = time.time()

        try:
            ret = run_tts(text, target_lang)
        except Exception as e:
            self.out_text.append(f"\n TTS error: {e}")
            return None

        if isinstance(ret, (str, Path)):
            p = Path(ret)
            return p if p.exists() else None

        time.sleep(0.1)
        after = self._snapshot_wavs()
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

    def _snapshot_wavs(self):
        cwd = Path.cwd()
        paths = list(cwd.glob("*.wav"))
        for sub in [p for p in cwd.iterdir() if p.is_dir()]:
            paths.extend(sub.glob("*.wav"))
        return paths


def main():
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
