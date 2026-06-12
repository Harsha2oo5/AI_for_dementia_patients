"""
camera_recognition_v2.py
------------------------
Upgraded live recognition loop with full unknown-person handling.

Controls during live session:
    Q        — quit
    R        — start on-the-fly registration of unknown person
               (press R, then SPACE 3 times to capture frames,
                then enter name/relation in terminal)
    SPACE    — capture frame (during registration mode)
    ESC      — cancel registration
    S        — save current frame for debugging

State machine:
    WATCHING  → normal recognition loop
    CAPTURING → collecting frames for new person registration
    DONE      → registration complete, back to WATCHING
"""

import cv2
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fsl_model import RelativeRecognitionModel
from new_person_handler import OpenSetRecogniser
from voice_output import VoiceSpeaker

PROTOTYPE_PATH     = "models/prototypes.pkl"
INFERENCE_INTERVAL = 2.5
DISPLAY_DURATION   = 4.0

# Overlay colours (BGR)
COL_KNOWN     = (80, 180, 80)     # green
COL_UNCERTAIN = (60, 160, 220)    # amber-ish
COL_UNKNOWN   = (60, 60, 220)     # red
COL_CAPTURE   = (200, 120, 40)    # blue


def draw_overlay(frame, result: dict, state: str, capture_count: int = 0):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 120), (w, h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    rec_state = result.get("state", "unknown")

    if state == "CAPTURING":
        colour = COL_CAPTURE
        cv2.putText(frame, f"REGISTRATION MODE — frames: {capture_count}/3",
                    (20, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)
        cv2.putText(frame, "Press SPACE to capture · ESC to cancel",
                    (20, h - 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(frame, "Position face clearly in the frame",
                    (20, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
        return

    if rec_state == "known":
        colour = COL_KNOWN
        cv2.putText(frame, result["name"],
                    (20, h - 88), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        cv2.putText(frame, result["relation"],
                    (20, h - 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
        cv2.putText(frame, f"{result['confidence']*100:.0f}% confidence",
                    (20, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    elif rec_state == "uncertain":
        colour = COL_UNCERTAIN
        cv2.putText(frame, f"Uncertain — might be {result.get('best_guess', '?')}",
                    (20, h - 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        cv2.putText(frame, "Not confident enough — please call a caregiver",
                    (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(frame, f"Best match score: {result['confidence']*100:.0f}%",
                    (20, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)

    else:  # unknown
        colour = COL_UNKNOWN
        cv2.putText(frame, "Person not recognised",
                    (20, h - 88), cv2.FONT_HERSHEY_SIMPLEX, 0.85, colour, 2)
        cv2.putText(frame, "Press R to register this person (caregiver)",
                    (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(frame, "Face saved — caregiver has been alerted",
                    (20, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)


def prompt_registration_details() -> tuple[str, str, str]:
    """Ask caregiver for name/relation/hint in terminal (non-blocking feel)."""
    print("\n--- Register new person ---")
    name     = input("Name: ").strip()
    relation = input("Relationship (e.g. Nurse, Neighbour, Friend): ").strip()
    hint     = input(f"Voice hint for patient (or press Enter for default): ").strip()
    if not hint:
        hint = f"{name} is your {relation}."
    return name, relation, hint


def main():
    print("[Camera v2] Loading model...")
    model = RelativeRecognitionModel()
    model.load_prototypes(PROTOTYPE_PATH)

    recogniser = OpenSetRecogniser(model)
    speaker    = VoiceSpeaker()

    # Check for pending unreviewed alerts from previous sessions
    pending = recogniser.pending_alerts()
    if pending:
        print(f"\n[Alert] {len(pending)} unregistered face(s) from previous sessions.")
        print(f"        Check: data/unknown_faces/  and  data/unknown_alerts.json")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Camera] Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\nControls:")
    print("  Q     = quit")
    print("  R     = register unknown person (caregiver)")
    print("  SPACE = capture frame during registration")
    print("  ESC   = cancel registration\n")

    app_state       = "WATCHING"   # or "CAPTURING"
    capture_count   = 0
    last_inference  = 0.0
    last_result     = {"state": "unknown", "matched": False, "name": None,
                       "relation": "", "hint": "", "confidence": 0.0, "all_scores": {}}
    result_shown_at = 0.0
    last_spoken     = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()

        # ---- Recognition (WATCHING state only) ----
        if app_state == "WATCHING" and (now - last_inference >= INFERENCE_INTERVAL):
            last_inference  = now
            result          = recogniser.recognise(frame)
            last_result     = result
            result_shown_at = now

            if result["state"] == "known" and result["name"] != last_spoken:
                speaker.speak(result["hint"])
                last_spoken = result["name"]

            elif result["state"] == "uncertain":
                last_spoken = None
                speaker.speak(result["hint"])

            elif result["state"] == "unknown":
                last_spoken = None
                recogniser.handle_unknown(frame, result)

        # ---- Draw overlay ----
        if app_state == "WATCHING" and (now - result_shown_at < DISPLAY_DURATION):
            draw_overlay(frame, last_result, app_state)
        elif app_state == "CAPTURING":
            draw_overlay(frame, {}, app_state, capture_count)

        # Status bar
        n_known = len(model.prototypes)
        status  = f"State: {app_state}  |  Registered: {n_known}  |  Q=quit  R=register"
        cv2.putText(frame, status, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Dementia Recognition System v2", frame)
        key = cv2.waitKey(1) & 0xFF

        # ---- Key handling ----
        if key == ord("q"):
            break

        elif key == ord("r") and app_state == "WATCHING":
            app_state     = "CAPTURING"
            capture_count = 0
            recogniser.start_capture()

        elif key == ord(" ") and app_state == "CAPTURING":
            capture_count = recogniser.add_frame(frame)
            if capture_count >= 3:
                # Got enough frames — ask caregiver for details
                cv2.destroyAllWindows()   # hide window while typing in terminal
                try:
                    name, relation, hint = prompt_registration_details()
                    summary = recogniser.finish_registration(name, relation, hint)
                    print(f"\nRegistered '{name}' ({summary['shots']}-shot). Confidence est: {summary['estimated_confidence']:.0%}")
                    speaker.speak(f"I have now learned to recognise {name}.")
                except Exception as e:
                    print(f"Registration failed: {e}")
                    recogniser.cancel_capture()
                app_state     = "WATCHING"
                capture_count = 0
                # Re-open display
                cap = cv2.VideoCapture(0)

        elif key == 27 and app_state == "CAPTURING":   # ESC
            recogniser.cancel_capture()
            app_state     = "WATCHING"
            capture_count = 0
            print("[Register] Cancelled.")

        elif key == ord("s"):
            cv2.imwrite("debug_frame.jpg", frame)
            print("[Debug] Saved frame as debug_frame.jpg")

    cap.release()
    cv2.destroyAllWindows()
    print("[Camera] Stopped.")


if __name__ == "__main__":
    main()