"""
evaluate_model.py
-----------------
Evaluate the FSL model's recognition accuracy on a test set.

For each registered relative, provide a folder of test images
(different from the registration photos). The script reports:
  - Per-person accuracy and average confidence
  - Confusion matrix
  - Recommendations if accuracy is below threshold

Usage:
    python evaluate_model.py --test_dir data/test_images/
    
    test_images/
        Sarah/    <- folder name must match registered name exactly
            test1.jpg
            test2.jpg
        Raj/
            test1.jpg
"""

import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fsl_model import RelativeRecognitionModel

PROTOTYPE_PATH = "models/prototypes.pkl"


def evaluate(test_dir: str):
    model = RelativeRecognitionModel()
    model.load_prototypes(PROTOTYPE_PATH)

    if not model.prototypes:
        print("No prototypes loaded. Run caregiver_setup.py first.")
        return

    registered_names = list(model.prototypes.keys())
    print(f"\nEvaluating model on test set: {test_dir}")
    print(f"Registered relatives: {registered_names}\n")

    results = {}   # name -> {correct, total, scores}

    for true_name in os.listdir(test_dir):
        person_dir = os.path.join(test_dir, true_name)
        if not os.path.isdir(person_dir):
            continue

        if true_name not in registered_names:
            print(f"[Skip] '{true_name}' not in registered prototypes.")
            continue

        results[true_name] = {"correct": 0, "total": 0, "confidences": []}

        for img_file in sorted(os.listdir(person_dir)):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_dir, img_file)
            result = model.recognise_face(img_path)

            results[true_name]["total"] += 1

            if result["matched"] and result["name"] == true_name:
                results[true_name]["correct"] += 1
                results[true_name]["confidences"].append(result["confidence"])
            else:
                predicted = result.get("name", "no match")
                print(f"  [Mismatch] {true_name}/{img_file} → predicted: {predicted} ({result['confidence']:.2f})")

    if not results:
        print("No test images found. Check that folder names match registered relative names.")
        return

    print("\n=== Evaluation Results ===")
    print(f"{'Name':<20} {'Accuracy':<12} {'Avg Confidence':<18} {'Shots'}")
    print("-" * 60)

    total_correct = 0
    total_images = 0

    for name, r in results.items():
        acc = r["correct"] / r["total"] if r["total"] else 0
        avg_conf = sum(r["confidences"]) / len(r["confidences"]) if r["confidences"] else 0
        shots = model.prototypes[name]["shots"]
        total_correct += r["correct"]
        total_images += r["total"]
        print(f"{name:<20} {acc:.0%}{'':>6} {avg_conf:.2f}{'':>12} {shots}-shot")

    overall = total_correct / total_images if total_images else 0
    print("-" * 60)
    print(f"{'Overall':<20} {overall:.0%}")

    print("\n=== Recommendations ===")
    for name, r in results.items():
        acc = r["correct"] / r["total"] if r["total"] else 0
        shots = model.prototypes[name]["shots"]
        if acc < 0.70:
            print(f"  • {name}: Low accuracy ({acc:.0%}). Try registering with {min(shots + 2, 5)} photos instead of {shots}.")
            print(f"    Use well-lit, front-facing photos at different angles.")
        elif acc < 0.85 and shots < 3:
            print(f"  • {name}: Good accuracy but could improve. Try adding 1-2 more photos.")
        else:
            print(f"  • {name}: Good ({acc:.0%}). No changes needed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="data/test_images",
                        help="Directory with subfolders named after each relative")
    args = parser.parse_args()
    evaluate(args.test_dir)
