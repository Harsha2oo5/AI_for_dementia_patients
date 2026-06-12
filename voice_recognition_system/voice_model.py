"""
voice_model.py
--------------
Real-time speaker recognition using voice embeddings.

Architecture (mirrors the face FSL model):
  Transfer Learning  : Resemblyzer (pretrained speaker encoder)
                       converts any audio clip → 256-d d-vector embedding.
  Few-Shot Learning  : Prototype per person = mean of their 1-5 voice samples.
                       At inference, cosine similarity finds the best match.

Usage:
    model = SpeakerRecognitionModel()
    model.register_speaker("Sarah", ["sarah_1.wav", "sarah_2.wav"], relation="Daughter")
    result = model.recognise_speaker("live_audio.wav")
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

THRESHOLD_KNOWN     = 0.75   # cosine similarity → confident match
THRESHOLD_UNCERTAIN = 0.50   # possible match


class SpeakerRecognitionModel:
    """
    Few-shot speaker identification model.

    Registers speakers with 1-5 voice samples.
    Uses pretrained Resemblyzer encoder (transfer learning).
    Identifies speakers via cosine similarity to stored prototypes.
    """

    def __init__(self):
        self.encoder    = None
        self.prototypes: dict = {}   # name → {embedding, relation, hint, shots}
        self._load_encoder()

    def _load_encoder(self):
        try:
            from resemblyzer import VoiceEncoder
            self.encoder = VoiceEncoder()
            print("[Voice Model] Resemblyzer encoder loaded.")
        except ImportError:
            print("[Voice Model] resemblyzer not installed. Run: pip install resemblyzer")
            self.encoder = None
        except Exception as e:
            print(f"[Voice Model] Encoder load failed: {e}")
            self.encoder = None

    # ── Registration ─────────────────────────────────────────────────────────

    def register_speaker(
        self,
        name: str,
        audio_paths: list,
        relation: str = "",
        hint: str = "",
    ) -> dict:
        """
        Register a speaker from 1-5 audio files.
        Computes mean embedding as the FSL prototype.
        """
        if not audio_paths:
            raise ValueError("At least one audio file required.")
        if self.encoder is None:
            raise RuntimeError("Voice encoder not loaded. Install resemblyzer.")

        embeddings = []
        failed     = []

        for path in audio_paths[:5]:
            emb = self._embed_audio(path)
            if emb is not None:
                embeddings.append(emb)
            else:
                failed.append(path)

        if not embeddings:
            raise RuntimeError(f"Could not embed any audio files for {name}.")
        if failed:
            print(f"[Voice Model] Skipped {len(failed)} file(s) for {name}.")

        prototype = np.mean(embeddings, axis=0)
        prototype = prototype / np.linalg.norm(prototype)

        self.prototypes[name] = {
            "embedding": prototype,
            "relation":  relation,
            "hint":      hint or f"{name} is your {relation}.",
            "shots":     len(embeddings),
        }

        est_conf = min(0.98, 0.68 + len(embeddings) * 0.07)
        print(f"[Voice Model] Registered '{name}' — {len(embeddings)}-shot, est. {est_conf:.0%}")

        return {
            "name":                 name,
            "shots":                len(embeddings),
            "relation":             relation,
            "estimated_confidence": est_conf,
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def recognise_speaker(self, audio_input) -> dict:
        """
        Identify who is speaking from an audio file or numpy array.

        Args:
            audio_input: file path (str) or numpy array (float32, 16kHz mono)

        Returns:
            {
                state:      "known" | "uncertain" | "unknown"
                matched:    bool
                name:       str | None
                relation:   str
                hint:       str
                confidence: float
                all_scores: {name: score}
                best_guess: str | None
            }
        """
        if not self.prototypes:
            return self._result("unknown", None, "", "No speakers registered yet.", 0.0, {})

        if self.encoder is None:
            return self._result("unknown", None, "", "Voice encoder not available.", 0.0, {})

        query_emb = self._embed_audio(audio_input)
        if query_emb is None:
            return self._result("unknown", None, "", "Could not process audio — too short or silent?", 0.0, {})

        # Cosine similarity against all prototypes
        scores = {}
        for name, data in self.prototypes.items():
            sim = float(np.dot(query_emb, data["embedding"]))
            scores[name] = round(sim, 4)

        best_name  = max(scores, key=scores.get)
        best_score = scores[best_name]
        best_data  = self.prototypes[best_name]

        if best_score >= THRESHOLD_KNOWN:
            return self._result(
                "known", best_name,
                best_data["relation"], best_data["hint"],
                best_score, scores,
            )
        elif best_score >= THRESHOLD_UNCERTAIN:
            return self._result(
                "uncertain", None,
                "", f"This might be {best_name}, but I am not sure. Please call a caregiver.",
                best_score, scores, best_guess=best_name,
            )
        else:
            return self._result("unknown", None, "", "Voice not recognised.", best_score, scores)

    def recognise_realtime(self, audio_array: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Convenience method for real-time audio arrays (from sounddevice).
        Normalises and passes to recognise_speaker.
        """
        if audio_array is None or len(audio_array) == 0:
            return self._result("unknown", None, "", "No audio data.", 0.0, {})

        # Flatten to mono float32
        audio = audio_array.flatten().astype(np.float32)

        # Trim silence (energy threshold)
        energy = np.sqrt(np.mean(audio ** 2))
        if energy < 0.001:
            return self._result("unknown", None, "", "Too quiet — please speak louder.", 0.0, {})

        return self.recognise_speaker(audio)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save_prototypes(self, path: str = "models/voice_prototypes.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.prototypes, f)
        print(f"[Voice Model] Saved {len(self.prototypes)} voice prototype(s) → {path}")

    def load_prototypes(self, path: str = "models/voice_prototypes.pkl"):
        if not os.path.exists(path):
            print(f"[Voice Model] No saved prototypes at {path}")
            return
        with open(path, "rb") as f:
            self.prototypes = pickle.load(f)
        print(f"[Voice Model] Loaded {len(self.prototypes)} voice prototype(s) from {path}")

    def list_speakers(self) -> list:
        return [
            {"name": n, "relation": d["relation"], "hint": d["hint"], "shots": d["shots"]}
            for n, d in self.prototypes.items()
        ]

    def remove_speaker(self, name: str):
        if name in self.prototypes:
            del self.prototypes[name]
            print(f"[Voice Model] Removed '{name}'.")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _embed_audio(self, audio_input) -> np.ndarray | None:
        """
        Convert audio to 256-d d-vector embedding using Resemblyzer.
        Accepts file path (str) or numpy array (float32).
        """
        try:
            from resemblyzer import preprocess_wav
            import numpy as np

            if isinstance(audio_input, str):
                if not os.path.exists(audio_input):
                    print(f"[Voice Model] File not found: {audio_input}")
                    return None
                wav = preprocess_wav(audio_input)
            elif isinstance(audio_input, np.ndarray):
                wav = preprocess_wav(audio_input)
            else:
                return None

            if len(wav) < 1600:   # less than 0.1s at 16kHz → skip
                return None

            emb = self.encoder.embed_utterance(wav)
            emb = emb / np.linalg.norm(emb)
            return emb

        except Exception as e:
            print(f"[Voice Model] Embed failed: {e}")
            return None
    def _confidence_label(self, score: float) -> str:
        if score >= 0.90:
            return "Very High"
        elif score >= 0.80:
            return "High"
        elif score >= 0.70:
            return "Medium"
        elif score >= 0.50:
            return "Low"
        return "Unknown"

    def _result(
        self, state, name, relation, hint, confidence, all_scores, best_guess=None
    ) -> dict:
        return {
            "state": state,
            "matched": state == "known",
            "name": name,
            "relation": relation,
            "hint": hint,
            "confidence": confidence,
            "confidence_label": self._confidence_label(confidence),
            "all_scores": all_scores,
            "best_guess": best_guess,
        }