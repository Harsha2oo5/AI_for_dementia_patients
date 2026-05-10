# Dementia AI — Relative Recognition System
### Adaptive AI using Few-Shot Learning + Transfer Learning

---

## Project structure

```
dementia_fsl/
│
├── models/
│   ├── fsl_model.py          ← Core ML model (FaceNet + prototypical network)
│   └── prototypes.pkl        ← Saved relative prototypes (created at runtime)
│
├── utils/
│   └── voice_output.py       ← Text-to-speech cognitive support hints
│
├── data/
│   └── relatives/            ← Put registration photos here
│       └── Sarah/
│           ├── photo1.jpg
│           └── photo2.jpg
│
├── caregiver_setup.py        ← Register relatives (caregiver runs this)
├── camera_recognition.py     ← Live recognition loop (patient device)
├── evaluate_model.py         ← Test accuracy on held-out images
└── requirements.txt
```

---

## Setup (VS Code / local Python)

### 1. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> FaceNet-PyTorch will automatically download the pretrained InceptionResNet-V1
> weights (~90 MB) on first run. No API key needed.

---

## How to use

### Step 1 — Register relatives (caregiver)

Run the interactive setup:

```bash
python caregiver_setup.py
```

You will be prompted to:
- Enter the person's name and relationship (e.g. "Sarah", "Daughter")
- Write a warm hint the patient will hear (e.g. "Sarah is your daughter, she visits every Sunday")
- Provide 1–5 photo paths

**Tips for good photos:**
- Use clear, well-lit, front-facing photos
- Include 1–2 photos at a slight angle for better generalisation
- 3 photos gives the best balance for few-shot learning

Or use the non-interactive CLI:

```bash
python caregiver_setup.py \
  --name "Sarah Patel" \
  --relation "Daughter" \
  --hint "Sarah is your daughter. She visits every Sunday and loves you very much." \
  --images data/relatives/sarah/photo1.jpg data/relatives/sarah/photo2.jpg
```

### Step 2 — Run live recognition (patient device)

```bash
python camera_recognition.py
```

- Opens the webcam
- Checks for known faces every 2.5 seconds
- Displays the person's name and relationship on screen
- Speaks the personalised hint aloud when a match is found
- Press **Q** to quit

### Step 3 — Evaluate accuracy (optional)

Prepare a `data/test_images/` folder:

```
data/test_images/
    Sarah/
        test1.jpg
        test2.jpg
    Raj/
        test1.jpg
```

Then run:

```bash
python evaluate_model.py --test_dir data/test_images/
```

This reports per-person accuracy and gives recommendations
(e.g. "add more photos for better performance").

---

## How the ML works

### Transfer Learning (FaceNet)
- FaceNet (InceptionResNet-V1) was pretrained on VGGFace2 — 3.3M images of 9,000+ people
- We reuse its learned face embeddings without retraining (fine-tuning is optional)
- Each face image → 512-dimensional embedding vector

### Few-Shot Learning (Prototypical Network)
- For each relative, all 1–5 photo embeddings are averaged into a single **prototype**
- At inference, the query embedding is compared to every prototype using **cosine similarity**
- Match is accepted if similarity ≥ 0.75 (adjustable in `fsl_model.py`)

### Why this works with small data
- FaceNet already understands faces (transfer learning handles the heavy lifting)
- Prototypical networks are designed for N-shot tasks — they generalise well from just 1 example
- Cosine similarity in embedding space is robust to lighting and angle variation

---

## Adjusting settings

In `models/fsl_model.py`:

```python
CONFIDENCE_THRESHOLD = 0.75   # lower → more matches (less strict)
                               # raise → fewer false positives
```

In `camera_recognition.py`:

```python
INFERENCE_INTERVAL = 2.5      # seconds between recognition attempts
DISPLAY_DURATION = 4.0        # how long the name overlay stays on screen
```

---

## Improving accuracy

| Problem | Solution |
|---|---|
| Low confidence | Add more photos (up to 5) |
| False positives | Raise CONFIDENCE_THRESHOLD to 0.82 |
| Misidentification | Add photos at different angles and lighting |
| No face detected | Ensure the face is well-lit and not obscured |

---

## Next steps / extensions

- **Flask web API** — expose `/register` and `/recognise` as REST endpoints
- **Streamlit caregiver dashboard** — web UI for caregivers to manage relatives
- **Fine-tuning** — finetune FaceNet on the registered relatives for higher accuracy
- **Age adaptation** — register both old and new photos; weight recent ones higher
