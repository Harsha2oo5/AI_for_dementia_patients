"""
incremental_fsl_model.py
-------------------------
Extends RelativeRecognitionModel with Incremental Few-Shot Learning.

PROBLEM IT SOLVES:
    Standard FSL is static — a prototype built from photos taken in 2024
    becomes less accurate as people age, change hairstyle, gain/lose weight,
    or change glasses. After 2-3 years confidence degrades noticeably.

SOLUTION — Incremental Prototype Update:
    Every time a person is recognised with HIGH confidence (>= 0.85),
    their new embedding is incorporated into the prototype using an
    Exponential Moving Average (EMA). The prototype gradually shifts
    toward the person's current appearance without any retraining.

    prototype_new = (1 - alpha) * prototype_old + alpha * new_embedding

    With alpha=0.05:
      - Each visit contributes 5% to the prototype
      - After ~20 confident visits the prototype fully reflects current appearance
      - Gradual drift — no sudden jumps even if one frame is slightly off

ADDITIONAL FEATURES:
    - Shot memory:    keeps last N embeddings as episodic memory
    - Drift detection: alerts if confidence drops suddenly (possible impersonation)
    - Rollback:       can revert to any historical prototype snapshot
    - Freeze mode:    disable updates for a specific person if needed

Usage:
    model = IncrementalFSLModel()
    model.load_prototypes("models/prototypes.pkl")

    # Recognition — auto-updates prototype on confident match
    result = model.recognise_and_update(frame)

    # Manual snapshot
    model.save_snapshot("models/snapshot_2025.pkl")
"""

import os
import sys
import pickle
import numpy as np
import datetime
import json
from pathlib import Path
from collections import deque

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fsl_model import RelativeRecognitionModel

# ── Incremental learning hyperparameters ─────────────────────────────────────
ALPHA_EMA          = 0.05   # EMA weight — how fast prototype adapts (0=never, 1=instant)
UPDATE_THRESHOLD   = 0.85   # minimum confidence to use a frame for prototype update
DRIFT_ALERT_DROP   = 0.15   # alert if confidence drops by this much vs historical average
MEMORY_SIZE        = 50     # keep last N embeddings in episodic memory per person
SNAPSHOT_INTERVAL  = 7      # auto-snapshot every N days

INCREMENTAL_META_PATH = os.path.join(_ROOT, "data", "incremental_meta.json")
SNAPSHOT_DIR          = os.path.join(_ROOT, "models", "snapshots")


class IncrementalFSLModel(RelativeRecognitionModel):
    """
    Drop-in replacement for RelativeRecognitionModel with incremental learning.

    Adds:
      - EMA prototype update on every confident recognition
      - Episodic memory buffer (last 50 embeddings per person)
      - Drift detection and alerting
      - Snapshot and rollback
      - Per-person freeze toggle
      - Confidence history tracking
    """

    def __init__(self, alpha: float = ALPHA_EMA, device: str = None):
        super().__init__(device=device)
        self.alpha = alpha

        # Episodic memory: {name: deque of (embedding, timestamp, confidence)}
        self._memory: dict[str, deque] = {}

        # Confidence history: {name: [float, ...]}
        self._conf_history: dict[str, list] = {}

        # Frozen persons — prototype will not be updated
        self._frozen: set = set()

        # Meta: update counts, last update time, drift alerts
        self._meta = self._load_meta()

        Path(SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════
    # CORE: RECOGNISE + AUTO-UPDATE
    # ═══════════════════════════════════════════════════════════════════════

    def recognise_and_update(self, image_input) -> dict:
        """
        Recognise a face AND automatically update the prototype if confident.

        This is the main method to call during live recognition.
        It replaces recognise_face() for the incremental learning mode.

        Returns the standard result dict PLUS:
            prototype_updated: bool — whether the prototype was updated
            drift_alert:       bool — whether a confidence drop was detected
            confidence_trend:  str  — "improving" | "stable" | "declining"
        """
        result = self.recognise_face(image_input)

        result["prototype_updated"] = False
        result["drift_alert"]       = False
        result["confidence_trend"]  = "stable"

        if not result.get("matched"):
            return result

        name       = result["name"]
        confidence = result["confidence"]

        # ── Get the embedding we just computed ───────────────────────────
        query_emb = self._embed_image(image_input)
        if query_emb is None:
            return result

        # ── Store in episodic memory ──────────────────────────────────────
        self._add_to_memory(name, query_emb, confidence)

        # ── Track confidence history ──────────────────────────────────────
        drift_alert = self._track_confidence(name, confidence)
        result["drift_alert"] = drift_alert
        result["confidence_trend"] = self._get_trend(name)

        # ── Update prototype if confidence high enough ────────────────────
        if confidence >= UPDATE_THRESHOLD and name not in self._frozen:
            self._ema_update(name, query_emb, confidence)
            result["prototype_updated"] = True
            self._meta["update_counts"][name] = \
                self._meta["update_counts"].get(name, 0) + 1
            self._meta["last_update"][name] = datetime.datetime.now().isoformat()
            self._save_meta()

        # ── Auto-snapshot check ───────────────────────────────────────────
        self._check_auto_snapshot()

        return result

    # ═══════════════════════════════════════════════════════════════════════
    # EMA PROTOTYPE UPDATE
    # ═══════════════════════════════════════════════════════════════════════

    def _ema_update(self, name: str, new_embedding: np.ndarray, confidence: float):
        """
        Exponential Moving Average update of the prototype.

        Formula:
            prototype_new = (1 - alpha) * prototype_old + alpha * new_embedding

        The alpha parameter controls adaptation speed:
            alpha = 0.05  → gentle, needs ~20 visits to fully adapt (recommended)
            alpha = 0.10  → moderate, needs ~10 visits
            alpha = 0.20  → fast, needs ~5 visits (risk: one bad frame can drift it)

        After update, re-normalise to keep the vector on the unit sphere.
        """
        old_proto = self.prototypes[name]["embedding"].copy()

        # Weighted EMA — higher confidence frames get slightly more weight
        effective_alpha = self.alpha * (0.5 + 0.5 * confidence)

        updated = (1.0 - effective_alpha) * old_proto + effective_alpha * new_embedding
        updated = updated / np.linalg.norm(updated)   # re-normalise

        self.prototypes[name]["embedding"] = updated

        print(f"[IncrementalFSL] Updated prototype for '{name}' "
              f"(alpha={effective_alpha:.3f}, conf={confidence:.3f}, "
              f"drift={self._cosine_drift(old_proto, updated):.4f})")

    def _cosine_drift(self, old: np.ndarray, new: np.ndarray) -> float:
        """How much did the prototype move? (0 = no change, 1 = complete change)"""
        return 1.0 - float(np.dot(old, new))

    # ═══════════════════════════════════════════════════════════════════════
    # EPISODIC MEMORY
    # ═══════════════════════════════════════════════════════════════════════

    def _add_to_memory(self, name: str, embedding: np.ndarray, confidence: float):
        if name not in self._memory:
            self._memory[name] = deque(maxlen=MEMORY_SIZE)
        self._memory[name].append({
            "embedding":  embedding,
            "confidence": confidence,
            "timestamp":  datetime.datetime.now().isoformat(),
        })

    def rebuild_prototype_from_memory(self, name: str, min_confidence: float = 0.80) -> bool:
        """
        Rebuild a person's prototype from their episodic memory.

        Useful if the prototype drifted in a bad direction (e.g. one bad frame
        with high confidence corrupted it slightly). Recomputes mean from the
        last N high-confidence embeddings instead of the EMA.

        Returns True if rebuilt successfully.
        """
        if name not in self._memory or name not in self.prototypes:
            print(f"[IncrementalFSL] No memory or prototype for '{name}'.")
            return False

        good_embeddings = [
            e["embedding"] for e in self._memory[name]
            if e["confidence"] >= min_confidence
        ]

        if not good_embeddings:
            print(f"[IncrementalFSL] No high-confidence frames in memory for '{name}'.")
            return False

        new_proto = np.mean(good_embeddings, axis=0)
        new_proto = new_proto / np.linalg.norm(new_proto)
        self.prototypes[name]["embedding"] = new_proto
        print(f"[IncrementalFSL] Rebuilt prototype for '{name}' "
              f"from {len(good_embeddings)} memory frames.")
        return True

    # ═══════════════════════════════════════════════════════════════════════
    # DRIFT DETECTION
    # ═══════════════════════════════════════════════════════════════════════

    def _track_confidence(self, name: str, confidence: float) -> bool:
        """
        Track confidence history and detect sudden drops.

        A sudden confidence drop could mean:
          - The person has significantly changed appearance (normal — update needed)
          - Someone different is being misidentified (security concern)

        Returns True if a drift alert should be raised.
        """
        if name not in self._conf_history:
            self._conf_history[name] = []

        self._conf_history[name].append(confidence)
        history = self._conf_history[name]

        # Need at least 5 data points
        if len(history) < 5:
            return False

        recent_avg    = np.mean(history[-5:])
        historical_avg = np.mean(history[:-5]) if len(history) > 5 else recent_avg
        drop           = historical_avg - recent_avg

        if drop >= DRIFT_ALERT_DROP:
            print(f"[DriftAlert] '{name}' confidence dropped by {drop:.2f} "
                  f"(was {historical_avg:.2f}, now {recent_avg:.2f})")
            return True

        return False

    def _get_trend(self, name: str) -> str:
        history = self._conf_history.get(name, [])
        if len(history) < 6:
            return "stable"
        recent = np.mean(history[-3:])
        older  = np.mean(history[-6:-3])
        delta  = recent - older
        if delta > 0.03:  return "improving"
        if delta < -0.03: return "declining"
        return "stable"

    def get_confidence_stats(self, name: str) -> dict:
        """Return confidence statistics for a registered person."""
        history = self._conf_history.get(name, [])
        if not history:
            return {"name": name, "n_recognitions": 0}
        return {
            "name":            name,
            "n_recognitions":  len(history),
            "avg_confidence":  float(np.mean(history)),
            "min_confidence":  float(np.min(history)),
            "max_confidence":  float(np.max(history)),
            "trend":           self._get_trend(name),
            "update_count":    self._meta["update_counts"].get(name, 0),
            "last_update":     self._meta["last_update"].get(name, "never"),
        }

    def get_all_stats(self) -> list:
        return [self.get_confidence_stats(name) for name in self.prototypes]

    # ═══════════════════════════════════════════════════════════════════════
    # FREEZE / UNFREEZE
    # ═══════════════════════════════════════════════════════════════════════

    def freeze(self, name: str):
        """Freeze a person's prototype — no further updates."""
        self._frozen.add(name)
        print(f"[IncrementalFSL] Prototype frozen for '{name}'.")

    def unfreeze(self, name: str):
        """Resume incremental updates for a person."""
        self._frozen.discard(name)
        print(f"[IncrementalFSL] Prototype unfrozen for '{name}'.")

    def is_frozen(self, name: str) -> bool:
        return name in self._frozen

    # ═══════════════════════════════════════════════════════════════════════
    # SNAPSHOTS AND ROLLBACK
    # ═══════════════════════════════════════════════════════════════════════

    def save_snapshot(self, label: str = "") -> str:
        """
        Save a snapshot of all current prototypes.
        Returns the snapshot file path.
        """
        ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        label = label.replace(" ", "_") or "auto"
        path  = os.path.join(SNAPSHOT_DIR, f"snapshot_{label}_{ts}.pkl")

        with open(path, "wb") as f:
            pickle.dump({
                "prototypes": self.prototypes,
                "timestamp":  datetime.datetime.now().isoformat(),
                "label":      label,
            }, f)

        print(f"[Snapshot] Saved → {path}")
        return path

    def rollback(self, snapshot_path: str) -> bool:
        """
        Restore prototypes from a snapshot.
        Use this if incremental updates caused incorrect drift.
        """
        if not os.path.exists(snapshot_path):
            print(f"[Rollback] Snapshot not found: {snapshot_path}")
            return False

        with open(snapshot_path, "rb") as f:
            snap = pickle.load(f)

        self.prototypes = snap["prototypes"]
        print(f"[Rollback] Restored snapshot from {snap['timestamp']} "
              f"({len(self.prototypes)} prototypes)")
        return True

    def list_snapshots(self) -> list:
        snaps = sorted(Path(SNAPSHOT_DIR).glob("snapshot_*.pkl"), reverse=True)
        result = []
        for s in snaps:
            try:
                with open(s, "rb") as f:
                    data = pickle.load(f)
                result.append({
                    "path":      str(s),
                    "timestamp": data.get("timestamp", ""),
                    "label":     data.get("label", ""),
                    "n_people":  len(data.get("prototypes", {})),
                })
            except Exception:
                pass
        return result

    def _check_auto_snapshot(self):
        """Auto-save snapshot every SNAPSHOT_INTERVAL days."""
        last_snap = self._meta.get("last_snapshot")
        if last_snap:
            days_since = (datetime.datetime.now() -
                          datetime.datetime.fromisoformat(last_snap)).days
            if days_since < SNAPSHOT_INTERVAL:
                return

        self.save_snapshot("auto")
        self._meta["last_snapshot"] = datetime.datetime.now().isoformat()
        self._save_meta()

    # ═══════════════════════════════════════════════════════════════════════
    # META PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════

    def _load_meta(self) -> dict:
        if os.path.exists(INCREMENTAL_META_PATH):
            try:
                with open(INCREMENTAL_META_PATH) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "update_counts": {},
            "last_update":   {},
            "last_snapshot": None,
        }

    def _save_meta(self):
        Path(os.path.dirname(INCREMENTAL_META_PATH)).mkdir(parents=True, exist_ok=True)
        with open(INCREMENTAL_META_PATH, "w") as f:
            json.dump(self._meta, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════════
    # OVERRIDE SAVE — include incremental metadata
    # ═══════════════════════════════════════════════════════════════════════

    def save_prototypes(self, path: str = "models/prototypes.pkl"):
        """Save prototypes + incremental metadata together."""
        super().save_prototypes(path)
        self._save_meta()