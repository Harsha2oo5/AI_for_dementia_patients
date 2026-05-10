"""
new_person_handler.py
---------------------
Handles the "new / unknown person" problem in FSL face recognition.

Three scenarios this module solves:

  SCENARIO 1 — Hard unknown (confidence < 0.50)
    Someone completely unknown walks in.
    Action: Save face crop, alert caregiver, do NOT guess.

  SCENARIO 2 — Uncertain match (0.50 <= confidence < 0.75)
    Looks a bit like someone but not confident enough.
    Action: Show "uncertain" overlay, ask patient to call caregiver.

  SCENARIO 3 — On-the-fly registration
    Caregiver is present and wants to register the new person
    immediately without stopping the camera session.
    Action: Capture N frames, build prototype, save — zero retraining.

This module plugs into camera_recognition.py as a drop-in upgrade.

Usage:
    from new_person_handler import OpenSetRecogniser
    recogniser = OpenSetRecogniser(model)
    result = recogniser.recognise(frame)
    recogniser.handle_unknown(frame, result)
"""

import os
import cv2
import time
import json
import datetime
import numpy as np
from pathlib import Path

# Thresholds — tune these for your use case
THRESHOLD_KNOWN     = 0.75   # >= this → confident match
THRESHOLD_UNCERTAIN = 0.50   # >= this but < KNOWN → uncertain
                             # < UNCERTAIN → hard unknown

UNKNOWN_SAVE_DIR    = "data/unknown_faces"
ALERT_LOG_PATH      = "data/unknown_alerts.json"
MIN_FRAMES_FOR_REG  = 3      # minimum frames captured for on-the-fly registration
COOLDOWN_SECONDS    = 10     # seconds before alerting about the same unknown face again


class OpenSetRecogniser:
    """
    Wraps RelativeRecognitionModel with open-set (unknown person) handling.

    Adds:
      - Three-tier confidence classification (known / uncertain / unknown)
      - Unknown face saving and caregiver alert logging
      - On-the-fly registration from live camera frames
      - Cooldown to avoid spamming alerts for the same face
    """

    def __init__(self, model):
        """
        Args:
            model: a RelativeRecognitionModel instance (already loaded)
        """
        self.model = model
        self._unknown_cooldown = {}     # face_hash -> last_alert_time
        self._pending_frames = []       # frames captured for on-the-fly registration

        Path(UNKNOWN_SAVE_DIR).mkdir(parents=True, exist_ok=True)
        Path("my_dataset").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main entry point — call this instead of model.recognise_face()
    # ------------------------------------------------------------------

    def recognise(self, frame: np.ndarray) -> dict:
        """
        Classify the face in the frame into one of three states:
            "known"     — confident match, return name/hint
            "uncertain" — possible match but not sure
            "unknown"   — person not in the system at all

        Returns extended result dict:
        {
            "state":      "known" | "uncertain" | "unknown",
            "matched":    bool,
            "name":       str | None,
            "relation":   str,
            "hint":       str,
            "confidence": float,
            "all_scores": dict,
            "best_guess": str | None,   # for uncertain state
        }
        """
        result = self.model.recognise_face(frame)
        confidence = result.get("confidence", 0.0)

        if not self.model.prototypes:
            return {**result, "state": "unknown", "best_guess": None,
                    "hint": "No relatives registered yet. Please ask a caregiver."}

        best_name = max(result["all_scores"], key=result["all_scores"].get) if result["all_scores"] else None

        if confidence >= THRESHOLD_KNOWN:
            return {**result, "state": "known", "best_guess": None}

        elif confidence >= THRESHOLD_UNCERTAIN:
            return {
                **result,
                "state":      "uncertain",
                "matched":    False,
                "name":       None,
                "best_guess": best_name,
                "hint":       f"This might be {best_name}, but I'm not sure. Please call for a caregiver.",
            }

        else:
            return {
                **result,
                "state":      "unknown",
                "matched":    False,
                "name":       None,
                "best_guess": None,
                "hint":       "This person is not recognised. Please ask a caregiver for help.",
            }

    # ------------------------------------------------------------------
    # Unknown face handling
    # ------------------------------------------------------------------

    def handle_unknown(self, frame: np.ndarray, result: dict) -> str | None:
        """
        Called when state == "unknown".
        Saves the face crop and logs an alert.
        Respects cooldown to avoid duplicate alerts.

        Returns:
            path to saved face image, or None if skipped (cooldown).
        """
        if result.get("state") != "unknown":
            return None

        face_crop = self._crop_face(frame)
        if face_crop is None:
            return None

        face_hash = self._hash_frame(face_crop)
        now = time.time()

        if face_hash in self._unknown_cooldown:
            elapsed = now - self._unknown_cooldown[face_hash]
            if elapsed < COOLDOWN_SECONDS:
                return None   # same face seen recently, skip alert

        self._unknown_cooldown[face_hash] = now

        # Save face crop with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(UNKNOWN_SAVE_DIR, f"unknown_{timestamp}.jpg")
        cv2.imwrite(save_path, face_crop)

        # Log alert to JSON file
        self._log_alert(save_path, result.get("confidence", 0.0))

        print(f"[Unknown] New face saved: {save_path}")
        print(f"[Unknown] Alert logged. Caregiver should check {ALERT_LOG_PATH}")

        return save_path

    # ------------------------------------------------------------------
    # On-the-fly registration (caregiver presses a key during live feed)
    # ------------------------------------------------------------------

    def start_capture(self):
        """Begin capturing frames for on-the-fly registration."""
        self._pending_frames = []
        print(f"[Register] Capturing frames... (need {MIN_FRAMES_FOR_REG})")

    def add_frame(self, frame: np.ndarray) -> int:
        """
        Add a frame to the pending registration buffer.
        Returns number of frames captured so far.
        """
        self._pending_frames.append(frame.copy())
        count = len(self._pending_frames)
        print(f"[Register] Frame {count}/{MIN_FRAMES_FOR_REG} captured.")
        return count

    def finish_registration(self, name: str, relation: str, hint: str = "") -> dict:
        """
        Complete registration using the captured frames.
        Saves face crops to disk and builds the FSL prototype.

        Args:
            name:     relative's name
            relation: relationship to patient
            hint:     voice hint text

        Returns:
            registration summary dict from the model.
        """
        if len(self._pending_frames) < 1:
            raise RuntimeError("No frames captured. Call add_frame() first.")

        # Save captured frames as images
        save_dir = os.path.join("data/relatives", name.replace(" ", "_"))
        os.makedirs(save_dir, exist_ok=True)

        saved_paths = []
        for i, frame in enumerate(self._pending_frames):
            path = os.path.join(save_dir, f"frame_{i+1}.jpg")
            cv2.imwrite(path, frame)
            saved_paths.append(path)

        print(f"[Register] Saved {len(saved_paths)} frames to {save_dir}")

        # Build prototype and register
        summary = self.model.register_relative(
            name,
            saved_paths,
            relation=relation,
            hint=hint or f"{name} is your {relation}.",
        )

        # Persist immediately
        self.model.save_prototypes("models/prototypes.pkl")
        self._pending_frames = []

        print(f"[Register] '{name}' registered instantly. No retraining needed.")
        return summary

    def cancel_capture(self):
        """Discard captured frames without registering."""
        self._pending_frames = []
        print("[Register] Capture cancelled.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _crop_face(self, frame: np.ndarray) -> np.ndarray | None:
        """Use MTCNN bounding box to crop face region from frame."""
        try:
            from PIL import Image
            img = Image.fromarray(frame[:, :, ::-1])   # BGR → RGB
            boxes, _ = self.model.detector.detect(img)
            if boxes is None or len(boxes) == 0:
                return None
            x1, y1, x2, y2 = [int(v) for v in boxes[0]]
            # Add 20px padding around face
            h, w = frame.shape[:2]
            x1 = max(0, x1 - 20)
            y1 = max(0, y1 - 20)
            x2 = min(w, x2 + 20)
            y2 = min(h, y2 + 20)
            return frame[y1:y2, x1:x2]
        except Exception as e:
            print(f"[Unknown] Face crop failed: {e}")
            return None

    def _hash_frame(self, frame: np.ndarray) -> str:
        """Simple perceptual hash to detect duplicate/same faces in cooldown."""
        small = cv2.resize(frame, (16, 16))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
        mean  = gray.mean()
        bits  = (gray > mean).flatten()
        return str(bits.tobytes())

    def _log_alert(self, image_path: str, confidence: float):
        """Append an unknown-face alert to the JSON log."""
        alerts = []
        if os.path.exists(ALERT_LOG_PATH):
            try:
                with open(ALERT_LOG_PATH) as f:
                    alerts = json.load(f)
            except Exception:
                alerts = []

        alerts.append({
            "timestamp":  datetime.datetime.now().isoformat(),
            "image_path": image_path,
            "confidence": round(confidence, 4),
            "reviewed":   False,
        })

        with open(ALERT_LOG_PATH, "w") as f:
            json.dump(alerts, f, indent=2)

    def pending_alerts(self) -> list[dict]:
        """Return all unreviewed unknown-face alerts."""
        if not os.path.exists(ALERT_LOG_PATH):
            return []
        try:
            with open(ALERT_LOG_PATH) as f:
                alerts = json.load(f)
            return [a for a in alerts if not a.get("reviewed")]
        except Exception:
            return []

    def mark_alert_reviewed(self, timestamp: str):
        """Mark an alert as reviewed (after caregiver registers the person)."""
        if not os.path.exists(ALERT_LOG_PATH):
            return
        with open(ALERT_LOG_PATH) as f:
            alerts = json.load(f)
        for a in alerts:
            if a["timestamp"] == timestamp:
                a["reviewed"] = True
        with open(ALERT_LOG_PATH, "w") as f:
            json.dump(alerts, f, indent=2)