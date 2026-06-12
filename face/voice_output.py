"""
voice_output.py
---------------
Text-to-speech cognitive support for dementia patients.

Uses pyttsx3 (offline, no API key needed) by default.
Falls back to gTTS (online, better quality) if preferred.

Usage:
    speaker = VoiceSpeaker()
    speaker.speak("Sarah is your daughter. She visits every Sunday.")
"""

import os
import threading


class VoiceSpeaker:
    """
    Converts recognition hints to calming spoken output.
    Designed to be gentle and slow — appropriate for dementia patients.
    """

    def __init__(self, use_gtts: bool = False, lang: str = "en"):
        self.use_gtts = use_gtts
        self.lang = lang
        self._engine = None

        if not use_gtts:
            self._init_pyttsx3()

    def _init_pyttsx3(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 140)      # slow, calm pace
            self._engine.setProperty("volume", 0.95)
            # prefer a female voice if available
            voices = self._engine.getProperty("voices")
            for v in voices:
                if "female" in v.name.lower() or "zira" in v.id.lower() or "karen" in v.id.lower():
                    self._engine.setProperty("voice", v.id)
                    break
        except Exception as e:
            print(f"[Voice] pyttsx3 init failed: {e}. Try: pip install pyttsx3")
            self._engine = None

    def speak(self, text: str, blocking: bool = False):
        """
        Speak the given text aloud.

        Args:
            text: the hint text to speak
            blocking: if True, wait until speech finishes before returning
        """
        print(f"[Voice] Speaking: {text}")
        if blocking:
            self._speak_now(text)
        else:
            t = threading.Thread(target=self._speak_now, args=(text,), daemon=True)
            t.start()

    def _speak_now(self, text: str):
        if self.use_gtts:
            self._speak_gtts(text)
        else:
            self._speak_pyttsx3(text)

    def _speak_pyttsx3(self, text: str):
        if self._engine is None:
            print(f"[Voice] No TTS engine available. Text: {text}")
            return
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as e:
            print(f"[Voice] pyttsx3 error: {e}")

    def _speak_gtts(self, text: str):
        try:
            from gtts import gTTS
            import tempfile
            import subprocess
            tts = gTTS(text=text, lang=self.lang, slow=True)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
            tts.save(tmp_path)
            # play with system player (cross-platform)
            if os.name == "nt":
                os.startfile(tmp_path)
            elif os.uname().sysname == "Darwin":
                subprocess.run(["afplay", tmp_path])
            else:
                subprocess.run(["mpg123", tmp_path], capture_output=True)
            os.unlink(tmp_path)
        except Exception as e:
            print(f"[Voice] gTTS error: {e}")

    def build_hint(self, name: str, relation: str, custom_hint: str = "") -> str:
        """
        Build a warm, patient-friendly spoken message.
        """
        if custom_hint:
            return custom_hint
        return f"This is {name}. They are your {relation}. They care about you very much."
