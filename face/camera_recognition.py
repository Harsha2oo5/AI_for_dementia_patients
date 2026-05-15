"""
camera_recognition_v2.py
------------------------
Live recognition loop with full unknown-person handling.

Run from inside the dementia_fsl/ folder:
    python camera_recognition_v2.py

Controls:
    Q      — quit
    R      — register unknown person (caregiver mode)
    SPACE  — capture frame during registration
    ESC    — cancel registration
    S      — save debug frame

States:
    WATCHING  — normal FSL recognition loop
    CAPTURING — collecting frames to register a new person
"""

import cv2
import time
import sys
import os
import numpy as np
# ── CRITICAL: add project root to path BEFORE any local imports ──
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fsl_model import RelativeRecognitionModel
from new_person_handler import OpenSetRecogniser
from voice_output import VoiceSpeaker

PROTOTYPE_PATH     = os.path.join(_ROOT, "models", "prototypes.pkl")
INFERENCE_INTERVAL = 2.5   # seconds between recognition attempts
DISPLAY_DURATION   = 4.0   # seconds to show result overlay

# BGR colours
COL_KNOWN     = (80, 200, 80)    # green
COL_UNCERTAIN = (40, 160, 220)   # blue-amber
COL_UNKNOWN   = (60,  60, 220)   # red
COL_CAPTURE   = (200, 140,  40)  # orange


# ── Overlay rendering ─────────────────────────────────────────────────────────

def draw_overlay(frame: "np.ndarray", result: dict, app_state: str, capture_count: int = 0):
    import cv2
    h, w = frame.shape[:2]

    # Semi-transparent bottom bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, h - 120), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(bar, 0.65, frame, 0.35, 0, frame)

    if app_state == "CAPTURING":
        cv2.putText(frame, f"REGISTER MODE — {capture_count}/3 frames",
                    (20, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_CAPTURE, 2)
        cv2.putText(frame, "SPACE = capture frame  |  ESC = cancel",
                    (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(frame, "Position face clearly — green box = ready",
                    (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
        return

    state = result.get("state", "unknown")

    if state == "known":
        cv2.putText(frame, result.get("name", ""),
                    (20, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        cv2.putText(frame, result.get("relation", ""),
                    (20, h - 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COL_KNOWN, 2)
        cv2.putText(frame, f"{result.get('confidence', 0)*100:.0f}% confidence",
                    (20, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    elif state == "uncertain":
        guess = result.get("best_guess", "?")
        cv2.putText(frame, f"Uncertain — might be {guess}",
                    (20, h - 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_UNCERTAIN, 2)
        cv2.putText(frame, "Not confident — please call a caregiver",
                    (20, h - 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(frame, f"Score: {result.get('confidence', 0)*100:.0f}%",
                    (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)

    else:  # unknown
        cv2.putText(frame, "Person not recognised",
                    (20, h - 88), cv2.FONT_HERSHEY_SIMPLEX, 0.85, COL_UNKNOWN, 2)
        cv2.putText(frame, "Press R to register this person (caregiver only)",
                    (20, h - 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(frame, "Face saved — caregiver alerted",
                    (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)


def draw_face_box(frame, recogniser, col_good=(80, 200, 80), col_bad=(60, 60, 220)):
    """Draw MTCNN face detection box on frame (live feedback)."""
    try:
        from PIL import Image
        import numpy as np
        img = Image.fromarray(frame[:, :, ::-1])
        boxes, probs = recogniser.model.detector.detect(img)
        if boxes is not None and len(boxes) > 0 and probs[0] is not None:
            conf = float(probs[0])
            x1, y1, x2, y2 = [int(v) for v in boxes[0]]
            col = col_good if conf >= 0.85 else col_bad
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, f"{conf*100:.0f}%",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            return conf >= 0.85
    except Exception:
        pass
    return False


def prompt_registration_details() -> tuple[str, str, str]:
    """Ask caregiver for name/relation/hint in terminal."""
    print("\n--- Register new person ---")
    name     = input("Name: ").strip()
    relation = input("Relationship (e.g. Nurse, Neighbour, Friend): ").strip()
    hint     = input("Voice hint for patient (press Enter for default): ").strip()
    if not hint:
        hint = f"{name} is your {relation}."
    return name, relation, hint


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print("[Camera v2] Starting — loading FSL model...")

    model = RelativeRecognitionModel()
    model.load_prototypes(PROTOTYPE_PATH)

    recogniser = OpenSetRecogniser(model)
    speaker    = VoiceSpeaker()

    pending = recogniser.pending_alerts()
    if pending:
        print(f"\n[Alert] {len(pending)} unreviewed unknown face(s) from previous sessions.")
       

    n = len(model.prototypes)
    if n == 0:
        print("\n[Warning] No relatives registered yet.")
        print("          Run: python caregiver_setup.py")
    else:
        print(f"\n[Ready] {n} relative(s) loaded: {list(model.prototypes.keys())}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Could not open webcam. Check camera is connected.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\nControls:  Q = quit  |  R = register  |  SPACE = capture  |  ESC = cancel reg")

    app_state       = "WATCHING"
    capture_count   = 0
    last_inference  = 0.0
    last_result     = {
        "state": "unknown", "matched": False, "name": None,
        "relation": "", "hint": "", "confidence": 0.0,
        "all_scores": {}, "best_guess": None,
    }
    result_shown_at = 0.0
    last_spoken     = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Camera] Frame read failed.")
            break

        now = time.time()

        # ── WATCHING: run FSL recognition on interval ──
        if app_state == "WATCHING" and (now - last_inference >= INFERENCE_INTERVAL):
            last_inference  = now
            result          = recogniser.recognise(frame)   # FSL + open-set
            last_result     = result
            result_shown_at = now

            if result["state"] == "known":
                if result["name"] != last_spoken:
                    speaker.speak(result["hint"])
                    last_spoken = result["name"]

            elif result["state"] == "uncertain":
                last_spoken = None
                speaker.speak(result["hint"])

            elif result["state"] == "unknown":
                last_spoken = None
                recogniser.handle_unknown(frame, result)   # save + alert

        # ── Draw face box (always) ──
        draw_face_box(frame, recogniser)

        # ── Draw result overlay ──
        if app_state == "WATCHING" and (now - result_shown_at < DISPLAY_DURATION):
            draw_overlay(frame, last_result, app_state)
        elif app_state == "CAPTURING":
            draw_overlay(frame, {}, app_state, capture_count)

        # ── Status bar ──
        status = (
            f"Mode: {app_state}  |  "
            f"Registered: {len(model.prototypes)}  |  "
            f"Q=quit  R=register  S=debug"
        )
        cv2.putText(frame, status, (14, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1)

        cv2.imshow("Dementia Recognition System v2", frame)
        key = cv2.waitKey(1) & 0xFF

        # ── Key handling ──
        if key == ord("q"):
            break

        elif key == ord("r") and app_state == "WATCHING":
            app_state     = "CAPTURING"
            capture_count = 0
            recogniser.start_capture()

        elif key == ord(" ") and app_state == "CAPTURING":
            capture_count = recogniser.add_frame(frame)
            if capture_count >= 3:
                # Close camera window while caregiver types in terminal
                cv2.destroyAllWindows()
                try:
                    name, relation, hint = prompt_registration_details()
                    summary = recogniser.finish_registration(name, relation, hint)
                    print(f"\n✓ Registered '{name}' — {summary['shots']}-shot · "
                          f"est. {summary['estimated_confidence']:.0%} confidence.")
                    speaker.speak(f"I have now learned to recognise {name}.")
                except Exception as e:
                    print(f"[Error] Registration failed: {e}")
                    recogniser.cancel_capture()

                app_state     = "WATCHING"
                capture_count = 0
                # Re-open camera
                cap.release()
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        elif key == 27 and app_state == "CAPTURING":    # ESC
            recogniser.cancel_capture()
            app_state     = "WATCHING"
            capture_count = 0

        elif key == ord("s"):
            cv2.imwrite("debug_frame.jpg", frame)
            print("[Debug] Saved debug_frame.jpg")

    cap.release()
    cv2.destroyAllWindows()
    print("[Camera] Stopped.")


if __name__ == "__main__":
    main()