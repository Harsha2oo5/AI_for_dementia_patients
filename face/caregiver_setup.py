"""
caregiver_setup.py
------------------
Caregiver tool to register relatives using LIVE CAMERA CAPTURE.

No photo paths needed — the caregiver simply looks at the camera.
The system captures 1-5 clear face shots automatically.

Run:
    python caregiver_setup.py

Or skip the menu and go straight to capturing:
    python caregiver_setup.py --name "Sarah" --relation "Daughter" --shots 3

Camera controls during capture:
    SPACE  — capture current frame manually
    A      — auto-capture (takes 3 frames automatically, 1 sec apart)
    R      — retake (discard all captured frames and start again)
    ENTER  — finish and register (once at least 1 frame captured)
    ESC    — cancel without saving
"""

import argparse
import os
import sys
import time
import cv2
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fsl_model import RelativeRecognitionModel

PROTOTYPE_PATH  = "models/prototypes.pkl"
CAPTURE_DIR     = "my_dataset/relatives"
MAX_SHOTS       = 5
AUTO_INTERVAL   = 1.2    # seconds between auto-captures
FACE_CONF_MIN   = 0.85   # MTCNN face confidence required to accept a frame


# ──────────────────────────────────────────────────────────────────────────────
# Live face capture
# ──────────────────────────────────────────────────────────────────────────────

def capture_face_shots(name: str, max_shots: int = 3) -> list[str]:
    """
    Open the webcam and let the caregiver capture face shots interactively.

    - Green box  = face detected clearly, ready to capture
    - Red box    = face found but not clear enough (move closer / better light)
    - No box     = no face found

    Returns list of saved image file paths (empty list if cancelled).
    """
    save_dir = os.path.join(CAPTURE_DIR, name.replace(" ", "_"))
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Camera] Could not open webcam. Check camera is connected.")
        return []

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Load MTCNN for real-time face detection feedback
    try:
        from facenet_pytorch import MTCNN
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        detector = MTCNN(keep_all=False, device=device, post_process=False)
        use_detector = True
    except Exception as e:
        print(f"[Warning] MTCNN not available ({e}). Face quality check disabled.")
        use_detector = False
        detector = None

    saved_paths = []
    auto_mode   = False
    last_auto   = 0.0
    flash_until = 0.0

    print(f"\n[Capture] Camera open — registering: {name}")
    print("  SPACE = capture manually")
    print("  A     = auto-capture")
    print("  R     = retake all")
    print("  ENTER = done & save")
    print("  ESC   = cancel\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        now     = time.time()
        h, w    = frame.shape[:2]

        # ── Detect face and draw bounding box ──
        face_detected = False
        if use_detector:
            try:
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray(frame[:, :, ::-1])
                boxes, probs = detector.detect(pil_img)
                if boxes is not None and len(boxes) > 0 and probs[0] is not None:
                    conf = float(probs[0])
                    x1, y1, x2, y2 = [int(v) for v in boxes[0]]
                    # Green = good, Red = poor quality
                    box_col = (80, 200, 80) if conf >= FACE_CONF_MIN else (60, 60, 220)
                    cv2.rectangle(display, (x1, y1), (x2, y2), box_col, 2)
                    cv2.putText(display, f"Face {conf*100:.0f}%",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_col, 1)
                    face_detected = conf >= FACE_CONF_MIN
            except Exception:
                pass
        else:
            face_detected = True   # no detector → always allow capture

        # ── Auto-capture logic ──
        if auto_mode and face_detected and (now - last_auto >= AUTO_INTERVAL):
            if len(saved_paths) < max_shots:
                path = _save_frame(frame, save_dir, len(saved_paths) + 1)
                saved_paths.append(path)
                flash_until = now + 0.3
                last_auto   = now
                print(f"  [Auto] Shot {len(saved_paths)}/{max_shots} captured.")
                if len(saved_paths) >= max_shots:
                    auto_mode = False
                    print("  [Auto] Complete. Press ENTER to save or R to retake.")

        # ── White flash on capture ──
        if now < flash_until:
            flash = display.copy()
            cv2.rectangle(flash, (0, 0), (w, h), (255, 255, 255), -1)
            cv2.addWeighted(flash, 0.3, display, 0.7, 0, display)

        # ── Draw HUD ──
        _draw_hud(display, name, saved_paths, max_shots, auto_mode, face_detected)

        cv2.imshow(f"Register: {name}", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:                           # ESC — cancel
            print("[Capture] Cancelled.")
            saved_paths = []
            break

        elif key in (13, 10):                   # ENTER — done
            if saved_paths:
                print(f"[Capture] {len(saved_paths)} shot(s) captured.")
                break
            else:
                print("[Capture] Capture at least 1 frame first (press SPACE or A).")

        elif key == ord(" "):                   # SPACE — manual capture
            if not face_detected and use_detector:
                print("[Capture] No clear face — adjust position or lighting.")
            elif len(saved_paths) < max_shots:
                path = _save_frame(frame, save_dir, len(saved_paths) + 1)
                saved_paths.append(path)
                flash_until = now + 0.3
                print(f"  Shot {len(saved_paths)}/{max_shots} captured.")
                if len(saved_paths) >= max_shots:
                    print("  Max shots reached. Press ENTER to save or R to retake.")
            else:
                print(f"  Already at {max_shots} shots. Press ENTER or R to retake.")

        elif key == ord("a"):                   # A — toggle auto-capture
            auto_mode = not auto_mode
            last_auto = 0.0
            print(f"[Capture] Auto-capture {'started' if auto_mode else 'stopped'}.")

        elif key == ord("r"):                   # R — retake
            saved_paths = []
            auto_mode   = False
            print("[Capture] All frames cleared. Start again.")

    cap.release()
    cv2.destroyAllWindows()
    return saved_paths


def _save_frame(frame, save_dir: str, index: int) -> str:
    """Save a captured frame to disk and return the path."""
    ts   = datetime.datetime.now().strftime("%H%M%S_%f")[:10]
    path = os.path.join(save_dir, f"shot_{index:02d}_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path


def _draw_hud(display, name: str, saved: list, max_shots: int, auto: bool, face_ok: bool):
    """Render the HUD overlay onto the camera feed."""
    h, w = display.shape[:2]

    # Semi-transparent bottom bar
    overlay = display.copy()
    cv2.rectangle(overlay, (0, h - 110), (w, h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

    # Shot indicator dots (filled green = captured, grey = empty)
    dot_start_x = 20
    for i in range(max_shots):
        colour = (80, 200, 80) if i < len(saved) else (70, 70, 70)
        cv2.circle(display, (dot_start_x + i * 30, h - 88), 10, colour, -1)
    cv2.putText(display,
                f"{len(saved)} / {max_shots} shots",
                (dot_start_x + max_shots * 30 + 12, h - 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1)

    # Status text
    if auto:
        s_text = "AUTO-CAPTURING — hold still..."
        s_col  = (60, 220, 220)
    elif face_ok:
        s_text = "Face ready  |  SPACE = capture   A = auto   ENTER = done"
        s_col  = (80, 210, 80)
    else:
        s_text = "No clear face — move closer or improve lighting"
        s_col  = (60, 80, 220)

    cv2.putText(display, s_text,   (20, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, s_col, 1)
    cv2.putText(display, f"Registering: {name}",
                (20, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(display, "R = retake   ESC = cancel",
                (w - 270, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1)


# ──────────────────────────────────────────────────────────────────────────────
# Menu actions
# ──────────────────────────────────────────────────────────────────────────────

def register_interactive(model: RelativeRecognitionModel):
    print("\n--- Register new relative ---")

    name = input("Full name (e.g. Sarah Patel): ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    relation = input("Relationship (e.g. Daughter, Son, Doctor): ").strip()
    hint     = input("Voice hint for patient (press Enter for auto): ").strip()
    if not hint:
        hint = f"{name} is your {relation}."

    shots_input = input("How many photos? (1–5, default 3): ").strip()
    try:
        shots = max(1, min(5, int(shots_input)))
    except ValueError:
        shots = 3

    print(f"\nOpening camera — position {name}'s face clearly in the frame.\n")
    image_paths = capture_face_shots(name, max_shots=shots)

    if not image_paths:
        print("No photos captured. Registration cancelled.")
        return

    print(f"\nBuilding FSL prototype from {len(image_paths)} photo(s)...")
    try:
        summary = model.register_relative(name, image_paths, relation=relation, hint=hint)
        model.save_prototypes(PROTOTYPE_PATH)
        print(f"\n✓ '{name}' registered successfully.")
        print(f"  Shots used      : {summary['shots']}")
        print(f"  Relation        : {summary['relation']}")
        print(f"  Est. confidence : {summary['estimated_confidence']:.0%}")
    except Exception as e:
        print(f"Error during registration: {e}")


def list_relatives(model: RelativeRecognitionModel):
    relatives = model.list_relatives()
    if not relatives:
        print("\nNo relatives registered yet.")
        return
    print(f"\n{len(relatives)} relative(s) registered:")
    for r in relatives:
        print(f"  • {r['name']} ({r['relation']}) — {r['shots']}-shot")
        print(f"    Hint: {r['hint']}")


def remove_relative(model: RelativeRecognitionModel):
    relatives = model.list_relatives()
    if not relatives:
        print("\nNo relatives registered.")
        return
    print("\nRegistered relatives:")
    for i, r in enumerate(relatives, 1):
        print(f"  {i}. {r['name']} ({r['relation']})")
    choice = input("Enter number to remove (or 0 to cancel): ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(relatives):
            model.remove_relative(relatives[idx]["name"])
            model.save_prototypes(PROTOTYPE_PATH)
            print(f"Removed '{relatives[idx]['name']}'.")
    except (ValueError, IndexError):
        print("Invalid selection.")


def interactive_mode(model: RelativeRecognitionModel):
    print("\n=== Dementia AI — Caregiver Setup (Camera) ===")
    while True:
        print("\nOptions:")
        print("  1. Register a new relative (live camera)")
        print("  2. List registered relatives")
        print("  3. Remove a relative")
        print("  4. Exit")
        choice = input("\nEnter choice (1–4): ").strip()

        if   choice == "1": register_interactive(model)
        elif choice == "2": list_relatives(model)
        elif choice == "3": remove_relative(model)
        elif choice == "4": break
        else: print("Please enter 1, 2, 3, or 4.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI (non-interactive, for scripting / automation)
# ──────────────────────────────────────────────────────────────────────────────

def cli_mode(args, model: RelativeRecognitionModel):
    print(f"\nCapturing {args.shots} photo(s) of {args.name}...")
    image_paths = capture_face_shots(args.name, max_shots=args.shots)
    if not image_paths:
        print("No photos captured.")
        sys.exit(1)
    hint = args.hint or f"{args.name} is your {args.relation}."
    summary = model.register_relative(args.name, image_paths, relation=args.relation, hint=hint)
    model.save_prototypes(PROTOTYPE_PATH)
    print(f"Registered '{args.name}' — {summary['shots']}-shot · est. {summary['estimated_confidence']:.0%} confidence.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register relatives via live camera.")
    parser.add_argument("--name",     type=str,            help="Relative's full name")
    parser.add_argument("--relation", type=str, default="", help="Relationship to patient")
    parser.add_argument("--hint",     type=str, default="", help="Voice hint text")
    parser.add_argument("--shots",    type=int, default=3,  help="Number of photos to capture (1–5)")
    args = parser.parse_args()

    model = RelativeRecognitionModel()
    model.load_prototypes(PROTOTYPE_PATH)

    if args.name:
        cli_mode(args, model)
    else:
        interactive_mode(model)