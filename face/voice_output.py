"""
voice_output.py
---------------
Text-to-speech for MindBridge AI — dementia patient support.

FOLDER: face_recogniton/voice_output.py

ENGINE PRIORITY (auto-detected):
  1. pyttsx3  — offline, needs: pip install pyttsx3
                Linux also needs: sudo apt install espeak espeak-data libespeak-dev
  2. gTTS     — online (needs internet), needs: pip install gtts
  3. espeak   — direct system call, Linux only, no pip needed if espeak installed
  4. print    — silent fallback (just prints to terminal)

THREAD SAFETY:
  pyttsx3 is NOT thread-safe — a Lock is used so only one speak() runs at a time.
  All speak() calls are non-blocking by default (run in daemon thread).

INSTALL (Ubuntu/Debian):
  sudo apt install espeak espeak-data libespeak-dev ffmpeg mpg123 -y
  pip install pyttsx3 gtts

Usage:
    speaker = VoiceSpeaker()
    speaker.speak("Sarah is your daughter. She visits every Sunday.")
    speaker.speak("Hi, may I know your name?")
"""

import os
import threading
import subprocess
import tempfile
from typing import Literal


# ── Engine type ────────────────────────────────────────────────────────────────
EngineType = Literal["pyttsx3", "gtts", "espeak", "none"]


class VoiceSpeaker:
    """
    Thread-safe TTS speaker with automatic engine detection and fallback.
    Designed for dementia care — slow, calm, clear voice.
    """

    def __init__(
        self,
        engine: EngineType = "auto",
        lang: str = "en",
        rate: int = 130,          # words per minute — slow and calm
        volume: float = 0.95,
    ):
        self.lang   = lang
        self.rate   = rate
        self.volume = volume

        self._engine_name: EngineType = "none"
        self._pyttsx3_engine = None
        self._lock = threading.Lock()   # pyttsx3 is single-threaded only

        if engine == "auto":
            self._engine_name = self._detect_engine()
        else:
            self._engine_name = engine

        if self._engine_name == "pyttsx3":
            self._init_pyttsx3()

        print(f"[Voice] Using engine: {self._engine_name}")

    # ── Engine detection ───────────────────────────────────────────────────────

    def _detect_engine(self) -> EngineType:
        # 1. Try pyttsx3
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.stop()
            return "pyttsx3"
        except Exception as e:
            print(f"[Voice] pyttsx3 unavailable: {e}")

        # 2. Try gTTS (needs internet + mpg123/ffmpeg to play)
        try:
            from gtts import gTTS
            player = self._find_audio_player()
            if player:
                return "gtts"
            else:
                print("[Voice] gTTS available but no audio player found "
                      "(install mpg123: sudo apt install mpg123)")
        except Exception as e:
            print(f"[Voice] gTTS unavailable: {e}")

        # 3. Try espeak directly
        if self._espeak_available():
            return "espeak"

        print("[Voice] ⚠ No TTS engine available — speech output disabled.")
        print("         To enable:")
        print("           sudo apt install espeak espeak-data libespeak-dev mpg123 -y")
        print("           pip install pyttsx3 gtts")
        return "none"

    def _find_audio_player(self) -> str | None:
        """Find an available audio player for MP3 files."""
        for player in ["mpg123", "mpg321", "ffplay", "cvlc", "aplay"]:
            try:
                result = subprocess.run(
                    ["which", player],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    return player
            except Exception:
                pass
        return None

    def _espeak_available(self) -> bool:
        try:
            result = subprocess.run(
                ["which", "espeak"],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except Exception:
            return False

    # ── pyttsx3 init ──────────────────────────────────────────────────────────

    def _init_pyttsx3(self):
        try:
            import pyttsx3
            self._pyttsx3_engine = pyttsx3.init()
            self._pyttsx3_engine.setProperty("rate",   self.rate)
            self._pyttsx3_engine.setProperty("volume", self.volume)

            # Prefer a clear female voice on all platforms
            voices = self._pyttsx3_engine.getProperty("voices")
            preferred_keywords = [
                "female", "zira", "karen", "victoria",
                "samantha", "tessa", "moira", "fiona",
                "en-gb", "en_gb", "en-us", "en_us",
            ]
            for keyword in preferred_keywords:
                for v in voices:
                    name_id = (v.name + v.id).lower()
                    if keyword in name_id:
                        self._pyttsx3_engine.setProperty("voice", v.id)
                        print(f"[Voice] Selected voice: {v.name}")
                        return

            print(f"[Voice] pyttsx3 ready ({len(voices)} voice(s) available).")

        except Exception as e:
            print(f"[Voice] pyttsx3 init error: {e}")
            print("        Linux fix: sudo apt install espeak espeak-data libespeak-dev -y")
            self._pyttsx3_engine = None
            self._engine_name    = "espeak" if self._espeak_available() else "none"

    # ── Public speak() ────────────────────────────────────────────────────────

    def speak(self, text: str, blocking: bool = False):
        """
        Speak text aloud. Non-blocking by default (daemon thread).

        Args:
            text     : text to speak
            blocking : if True, wait for speech to finish before returning
        """
        if not text or not text.strip():
            return

        print(f"[Voice] 🔊 {text}")

        if blocking:
            self._speak_now(text)
        else:
            t = threading.Thread(target=self._speak_now,
                                 args=(text,), daemon=True)
            t.start()

    def _speak_now(self, text: str):
        """Route to the correct engine — always called inside a thread."""
        with self._lock:   # ensure only one speak at a time
            if self._engine_name == "pyttsx3":
                self._speak_pyttsx3(text)
            elif self._engine_name == "gtts":
                self._speak_gtts(text)
            elif self._engine_name == "espeak":
                self._speak_espeak(text)
            else:
                pass   # silent — already printed above

    # ── pyttsx3 ───────────────────────────────────────────────────────────────

    def _speak_pyttsx3(self, text: str):
        if self._pyttsx3_engine is None:
            self._init_pyttsx3()
        if self._pyttsx3_engine is None:
            self._speak_espeak(text)   # fallback
            return
        try:
            self._pyttsx3_engine.say(text)
            self._pyttsx3_engine.runAndWait()
        except RuntimeError as e:
            # "run loop already started" — reinitialise engine
            print(f"[Voice] pyttsx3 RuntimeError: {e} — reinitialising…")
            try:
                import pyttsx3
                self._pyttsx3_engine = pyttsx3.init()
                self._pyttsx3_engine.setProperty("rate",   self.rate)
                self._pyttsx3_engine.setProperty("volume", self.volume)
                self._pyttsx3_engine.say(text)
                self._pyttsx3_engine.runAndWait()
            except Exception as e2:
                print(f"[Voice] pyttsx3 reinit failed: {e2} — falling back to espeak")
                self._speak_espeak(text)
        except Exception as e:
            print(f"[Voice] pyttsx3 error: {e}")
            self._speak_espeak(text)   # fallback

    # ── gTTS ──────────────────────────────────────────────────────────────────

    def _speak_gtts(self, text: str):
        try:
            from gtts import gTTS

            tts = gTTS(text=text, lang=self.lang, slow=True)

            with tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False
            ) as f:
                tmp_path = f.name

            tts.save(tmp_path)

            player = self._find_audio_player()
            if player:
                subprocess.run(
                    [player, tmp_path],
                    capture_output=True
                )
            elif os.name == "nt":
                os.startfile(tmp_path)
            elif hasattr(os, "uname") and os.uname().sysname == "Darwin":
                subprocess.run(["afplay", tmp_path])
            else:
                print(f"[Voice] gTTS: no player found for {tmp_path}")

            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        except Exception as e:
            print(f"[Voice] gTTS error: {e}")
            self._speak_espeak(text)   # fallback

    # ── espeak (direct system call) ───────────────────────────────────────────

    def _speak_espeak(self, text: str):
        """
        Direct espeak call — works on Linux without any Python package.
        Install: sudo apt install espeak -y
        """
        try:
            # -s = speed (words/min), -a = amplitude, -v = voice
            subprocess.run(
                [
                    "espeak",
                    "-s", str(self.rate),
                    "-a", "180",
                    "-v", f"{self.lang}+f3",   # female voice variant
                    text,
                ],
                capture_output=True,
                timeout=30,
            )
        except FileNotFoundError:
            print("[Voice] espeak not found. Install: sudo apt install espeak -y")
        except Exception as e:
            print(f"[Voice] espeak error: {e}")

    # ── Utility ───────────────────────────────────────────────────────────────

    def build_hint(
        self,
        name: str,
        relation: str,
        custom_hint: str = "",
        gender: str = "they",   # "he", "she", or "they"
    ) -> str:
        """
        Build a warm, patient-friendly spoken message.

        Args:
            name        : person's name
            relation    : relationship to patient
            custom_hint : override text if provided
            gender      : "he", "she", or "they"
        """
        if custom_hint:
            return custom_hint

        pronoun = {
            "he":   ("He",   "his"),
            "she":  ("She",  "her"),
            "they": ("They", "their"),
        }.get(gender.lower(), ("They", "their"))

        return (
            f"This is {name}. "
            f"{pronoun[0]} is your {relation}. "
            f"{pronoun[0]} cares about you very much."
        )

    def test(self):
        """Quick test to verify TTS is working."""
        self.speak(
            "Hello. This is MindBridge AI. Voice output is working correctly.",
            blocking=True,
        )

    @property
    def engine_name(self) -> str:
        return self._engine_name

    @property
    def is_available(self) -> bool:
        return self._engine_name != "none"