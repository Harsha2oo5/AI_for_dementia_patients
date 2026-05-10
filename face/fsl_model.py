"""
fsl_model.py
------------
Core ML model for relative recognition.

Architecture:
  - Transfer Learning : Pretrained FaceNet (InceptionResNet-V1)
                        extracts a 512-d embedding per face image.
  - Few-Shot Learning : Prototypical Network computes a class prototype
                        (mean embedding) from 1-5 registered photos.
                        At inference, cosine similarity is computed
                        between the live-camera embedding and all
                        stored prototypes. The closest match above
                        a confidence threshold is returned.

Usage:
  model = RelativeRecognitionModel()
  model.register_relative("Sarah", ["img1.jpg", "img2.jpg"], relation="Daughter", hint="...")
  result = model.recognise_face("live_frame.jpg")
"""

import os
import json
import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1


class RelativeRecognitionModel:
    """
    Few-shot face recognition model for dementia patients.

    Registers relatives with 1-5 photos (few-shot).
    Uses pretrained FaceNet for face embeddings (transfer learning).
    Recognises faces via cosine similarity to class prototypes.
    """

    CONFIDENCE_THRESHOLD = 0.75   # minimum cosine similarity for a valid match
    IMG_SIZE = 160                 # FaceNet input size

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Model] Using device: {self.device}")

        # MTCNN: face detector (handles alignment automatically)
        self.detector = MTCNN(
            image_size=self.IMG_SIZE,
            margin=20,
            keep_all=False,
            device=self.device,
        )

        # FaceNet: pretrained on VGGFace2 — Transfer Learning backbone
        self.encoder = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # Prototype store: {name -> {"embedding": np.array, "relation": str, "hint": str, "shots": int}}
        self.prototypes: dict = {}

    # ------------------------------------------------------------------
    # Registration (caregiver side)
    # ------------------------------------------------------------------

    def register_relative(
        self,
        name: str,
        image_paths: list[str],
        relation: str = "",
        hint: str = "",
    ) -> dict:
        """
        Register a relative from 1-5 photos.
        Computes one prototype embedding (mean of all shot embeddings).

        Returns summary dict with name, shots used, confidence estimate.
        """
        if not image_paths:
            raise ValueError("At least one image is required.")

        embeddings = []
        failed = []

        for path in image_paths:
            emb = self._embed_image(path)
            if emb is not None:
                embeddings.append(emb)
            else:
                failed.append(path)

        if not embeddings:
            raise RuntimeError(f"Could not detect a face in any of the provided images for {name}.")

        if failed:
            print(f"[Warning] No face detected in {len(failed)} image(s) for {name}, skipping.")

        # Prototype = mean of all valid embeddings (L2-normalised)
        prototype = np.mean(embeddings, axis=0)
        prototype = prototype / np.linalg.norm(prototype)

        self.prototypes[name] = {
            "embedding": prototype,
            "relation": relation,
            "hint": hint or f"{name} is your {relation}.",
            "shots": len(embeddings),
        }

        confidence_estimate = min(0.98, 0.70 + len(embeddings) * 0.06)
        print(f"[Model] Registered '{name}' with {len(embeddings)}-shot prototype. Est. confidence: {confidence_estimate:.0%}")

        return {
            "name": name,
            "shots": len(embeddings),
            "relation": relation,
            "estimated_confidence": confidence_estimate,
        }

    # ------------------------------------------------------------------
    # Inference (patient side / live camera)
    # ------------------------------------------------------------------

    def recognise_face(self, image_input) -> dict:
        """
        Identify who is in the image.

        Args:
            image_input: file path (str) or PIL Image or numpy array (BGR from OpenCV)

        Returns dict:
            {
                "matched": bool,
                "name": str or None,
                "relation": str,
                "hint": str,
                "confidence": float,
                "all_scores": {name: score, ...}
            }
        """
        if not self.prototypes:
            return {"matched": False, "name": None, "relation": "", "hint": "No relatives registered yet.", "confidence": 0.0, "all_scores": {}}

        query_emb = self._embed_image(image_input)
        if query_emb is None:
            return {"matched": False, "name": None, "relation": "", "hint": "No face detected in frame.", "confidence": 0.0, "all_scores": {}}

        # Cosine similarity against every prototype
        scores = {}
        for name, data in self.prototypes.items():
            sim = float(np.dot(query_emb, data["embedding"]))
            scores[name] = round(sim, 4)

        best_name = max(scores, key=scores.get)
        best_score = scores[best_name]

        if best_score >= self.CONFIDENCE_THRESHOLD:
            info = self.prototypes[best_name]
            return {
                "matched": True,
                "name": best_name,
                "relation": info["relation"],
                "hint": info["hint"],
                "confidence": best_score,
                "all_scores": scores,
            }
        else:
            return {
                "matched": False,
                "name": None,
                "relation": "",
                "hint": "This person was not recognised. Please ask a caregiver for help.",
                "confidence": best_score,
                "all_scores": scores,
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_prototypes(self, path: str = "models/prototypes.pkl"):
        """Save registered prototypes to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.prototypes, f)
        print(f"[Model] Prototypes saved to {path}")

    def load_prototypes(self, path: str = "models/prototypes.pkl"):
        """Load registered prototypes from disk."""
        if not os.path.exists(path):
            print(f"[Model] No saved prototypes found at {path}")
            return
        with open(path, "rb") as f:
            self.prototypes = pickle.load(f)
        print(f"[Model] Loaded {len(self.prototypes)} prototypes from {path}")

    def list_relatives(self) -> list[dict]:
        """Return a list of all registered relatives (without embeddings)."""
        return [
            {
                "name": name,
                "relation": data["relation"],
                "hint": data["hint"],
                "shots": data["shots"],
            }
            for name, data in self.prototypes.items()
        ]

    def remove_relative(self, name: str):
        """Remove a registered relative."""
        if name in self.prototypes:
            del self.prototypes[name]
            print(f"[Model] Removed '{name}' from prototypes.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_image(self, image_input) -> np.ndarray | None:
        """
        Detect face, align, and extract 512-d FaceNet embedding.
        Returns L2-normalised numpy array or None if no face found.
        """
        img = self._load_image(image_input)
        if img is None:
            return None

        # MTCNN: detect + align face → (1, 3, 160, 160) tensor
        face_tensor = self.detector(img)
        if face_tensor is None:
            return None

        face_tensor = face_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.encoder(face_tensor)

        emb = embedding.squeeze().cpu().numpy()
        emb = emb / np.linalg.norm(emb)       # L2 normalise
        return emb

    def _load_image(self, image_input) -> Image.Image | None:
        """Convert various input types to PIL RGB Image."""
        try:
            if isinstance(image_input, str):
                return Image.open(image_input).convert("RGB")
            elif isinstance(image_input, np.ndarray):
                # OpenCV BGR → RGB
                return Image.fromarray(image_input[:, :, ::-1])
            elif isinstance(image_input, Image.Image):
                return image_input.convert("RGB")
            else:
                print(f"[Model] Unknown image type: {type(image_input)}")
                return None
        except Exception as e:
            print(f"[Model] Failed to load image: {e}")
            return None
