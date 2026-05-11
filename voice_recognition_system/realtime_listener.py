"""
realtime_listener.py
---------------------
Continuous real-time voice recognition.

Listens to the microphone in chunks. For each chunk:
  1. Checks if someone is speaking (VAD — voice activity detection)
  2. Buffers audio until speech ends
  3. Runs speaker identification (who is speaking)
  4. Runs Whisper transcription (what they said)
  5. Returns result via callback or queue

Usage (standalone):
    python realtime_listener.py

Usage (in Streamlit / threaded):
    from realtime_listener import RealtimeListener
    listener = RealtimeListener(voice_model, on_result=my_callback)
    listener.start()
    ...
    listener.stop()
"""

import os
import sys
import time
import queue
import threading
import numpy as np
import sounddevice as sd
import datetime
import wave

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Audio settings ────────────────────────────────────────────────────────────
SAMPLE_RATE     = 16000   # Hz — Resemblyzer and Whisper both expect 16kHz
CHUNK_DURATION  = 0.5     # seconds per audio chunk
CHUNK_SAMPLES   = int(SAMPLE_RATE * CHUNK_DURATION)
SPEECH_THRESHOLD = 0.012  # RMS energy threshold for voice activity detection
MIN_SPEECH_SECS  = 1.2    # minimum buffered speech before running recognition
MAX_SPEECH_SECS  = 6.0    # maximum buffer before forcing recognition
SILENCE_TIMEOUT  = 1.0    # seconds of silence after speech ends recognition run


class RealtimeListener:
    """
    Continuous microphone listener with real-time speaker ID + transcription.

    Uses a producer-consumer architecture:
      - Audio thread  : records chunks from microphone
      - VAD thread    : detects speech start/end, buffers audio
      - Result queue  : completed utterances ready for recognition
    """

    def __init__(self, voice_model, on_result=None, save_dir=None):
        """
        Args:
            voice_model : SpeakerRecognitionModel instance
            on_result   : callback(result_dict) called after each utterance
            save_dir    : if set, saves each utterance as WAV for debugging
        """
        self.voice_model  = voice_model
        self.on_result    = on_result
        self.save_dir     = save_dir
        self._stop_event  = threading.Event()
        self._audio_queue = queue.Queue()
        self._result_queue = queue.Queue(maxsize=50)
        self._thread_audio = None
        self._thread_vad   = None
        self._thread_recog = None
        self.is_running    = False
        self.last_result   = None

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def start(self):
        """Start listening in background threads."""
        if self.is_running:
            return
        self._stop_event.clear()
        self.is_running = True

        self._thread_vad   = threading.Thread(target=self._vad_loop,   daemon=True)
        self._thread_recog = threading.Thread(target=self._recog_loop, daemon=True)

        self._thread_vad.start()
        self._thread_recog.start()
        print("[Listener] Started — listening for speech.")

    def stop(self):
        """Stop all threads."""
        self._stop_event.set()
        self.is_running = False
        print("[Listener] Stopped.")

    def get_latest_result(self) -> dict | None:
        """Non-blocking — returns latest result or None if nothing new."""
        try:
            result = None
            while not self._result_queue.empty():
                result = self._result_queue.get_nowait()
            if result:
                self.last_result = result
            return result
        except queue.Empty:
            return None

    # ── VAD loop (voice activity detection + buffering) ───────────────────────

    def _vad_loop(self):
        """
        Reads microphone in chunks. Detects speech start/end.
        When a complete utterance is captured, puts it in the audio queue.
        """
        speech_buffer  = []
        speech_active  = False
        silence_start  = None
        speech_start   = None

        def audio_callback(indata, frames, time_info, status):
            chunk = indata[:, 0].copy()   # mono
            self._audio_queue.put(chunk)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=audio_callback,
        ):
            while not self._stop_event.is_set():
                try:
                    chunk = self._audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                rms = float(np.sqrt(np.mean(chunk ** 2)))
                now = time.time()

                if rms > SPEECH_THRESHOLD:
                    if not speech_active:
                        speech_active = True
                        speech_start  = now
                        speech_buffer = []
                        print("[VAD] Speech started.")
                    speech_buffer.append(chunk)
                    silence_start = None

                elif speech_active:
                    speech_buffer.append(chunk)   # buffer a bit of trailing silence
                    if silence_start is None:
                        silence_start = now

                    buffered_secs = len(speech_buffer) * CHUNK_DURATION
                    silent_secs   = now - silence_start

                    should_flush = (
                        silent_secs  >= SILENCE_TIMEOUT or
                        buffered_secs >= MAX_SPEECH_SECS
                    )

                    if should_flush and buffered_secs >= MIN_SPEECH_SECS:
                        audio = np.concatenate(speech_buffer)
                        self._process_utterance(audio)
                        speech_buffer = []
                        speech_active = False
                        silence_start = None
                        print("[VAD] Speech ended — utterance queued.")

                    elif should_flush:
                        # too short — discard
                        speech_buffer = []
                        speech_active = False

    # ── Utterance processing ─────────────────────────────────────────────────

    def _process_utterance(self, audio: np.ndarray):
        """Optionally save audio then queue for recognition."""
        save_path = None
        if self.save_dir:
            ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:18]
            save_path = os.path.join(self.save_dir, f"utterance_{ts}.wav")
            _save_wav(audio, save_path, SAMPLE_RATE)

        self._audio_queue.put({"audio": audio, "path": save_path})

    # ── Recognition loop ─────────────────────────────────────────────────────

    def _recog_loop(self):
        """
        Picks up buffered utterances, runs speaker ID + Whisper transcription.
        """
        while not self._stop_event.is_set():
            try:
                item = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if not isinstance(item, dict):
                continue   # raw chunk from VAD init, skip

            audio     = item["audio"]
            save_path = item.get("path")

            # ── Speaker identification ──
            speaker_result = self.voice_model.recognise_realtime(audio, SAMPLE_RATE)

            # ── Whisper transcription ──
            transcript = ""
            try:
                transcript = _transcribe(audio, save_path)
            except Exception as e:
                transcript = f"[transcription error: {e}]"

            result = {
                **speaker_result,
                "transcript":  transcript,
                "timestamp":   datetime.datetime.now().isoformat(),
                "audio_path":  save_path,
            }

            self._result_queue.put(result)
            self.last_result = result

            if self.on_result:
                try:
                    self.on_result(result)
                except Exception as e:
                    print(f"[Listener] on_result callback error: {e}")


# ── Whisper transcription ─────────────────────────────────────────────────────

_whisper_model = None

def _transcribe(audio: np.ndarray, wav_path: str | None = None) -> str:
    """Transcribe audio using Whisper. Loads model once and caches it."""
    global _whisper_model
    try:
        import whisper
        if _whisper_model is None:
            print("[Whisper] Loading base model (first time)…")
            _whisper_model = whisper.load_model("base")

        if wav_path and os.path.exists(wav_path):
            result = _whisper_model.transcribe(wav_path, fp16=False)
        else:
            # transcribe from array directly
            audio_norm = audio.astype(np.float32)
            audio_norm = audio_norm / max(np.abs(audio_norm).max(), 1e-8)
            result = _whisper_model.transcribe(audio_norm, fp16=False)

        return result.get("text", "").strip()

    except ImportError:
        return "[Install openai-whisper for transcription]"
    except Exception as e:
        return f"[transcription failed: {e}]"


def _save_wav(audio: np.ndarray, path: str, sample_rate: int = 16000):
    """Save float32 numpy array as 16-bit WAV."""
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from voice_model import SpeakerRecognitionModel

    vm = SpeakerRecognitionModel()
    vm.load_prototypes("models/voice_prototypes.pkl")

    if not vm.prototypes:
        print("No speakers registered. Run caregiver_voice_setup.py first.")
        sys.exit(1)

    print(f"Registered speakers: {list(vm.prototypes.keys())}")
    print("Listening... Speak into the microphone. Ctrl+C to stop.\n")

    def on_result(r):
        state = r["state"]
        name  = r.get("name") or r.get("best_guess") or "unknown"
        conf  = r["confidence"]
        text  = r.get("transcript","")
        print(f"\n[{state.upper()}] {name} ({conf*100:.0f}%) — \"{text}\"")

    listener = RealtimeListener(vm, on_result=on_result,
                                 save_dir="data/recordings")
    listener.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        listener.stop()