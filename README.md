# AI for Dementia Patients

### AI-Powered Memory Assistance System using Face Recognition, Speaker Recognition, Few-Shot Learning, and Explainable AI

---

## Overview

Dementia often causes memory loss, difficulty recognizing family members, and challenges in recalling important personal information. This project aims to improve the quality of life for dementia patients by leveraging Artificial Intelligence, Computer Vision, and Machine Learning to provide real-time memory assistance.

The system combines facial recognition, speaker recognition, contextual memory retrieval, caregiver support, and explainable AI techniques to help patients identify familiar individuals and receive personalized cognitive assistance during daily interactions.

---

## Key Features

### Face Recognition

* CNN-based facial recognition system
* FaceNet-powered transfer learning
* Few-Shot Learning for recognizing new individuals with limited samples
* Open-set recognition for handling unknown individuals
* Real-time webcam-based inference

### Speaker Recognition

* MFCC-based voice feature extraction
* Speaker embedding generation and matching
* Incremental speaker registration
* Voice-based identity verification

### Memory Assistance

* Personalized contextual memory retrieval
* Relationship-aware cognitive reminders
* Real-time assistance for recognizing relatives and caregivers

### Caregiver Support

* Unknown-person detection and alerts
* New-person registration workflow
* Confidence-based safety mechanisms
* Caregiver notification support

### Explainable AI

* GradCAM visualizations
* Confidence-aware decision making
* Transparent prediction explanations

### User Interface

* Interactive Streamlit dashboard
* Real-time recognition display
* User-friendly dementia assistance interface

---

## System Architecture

```text
                    +----------------------+
                    |    Camera Input      |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Face Recognition     |
                    | FaceNet + FSL Model  |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Identity Prediction  |
                    +----------+-----------+
                               |
              +----------------+----------------+
              |                                 |
              v                                 v

        Known Person                    Unknown Person
              |                                 |
              v                                 v

   Context Retrieval             Caregiver Notification
              |
              v

      Memory Assistance


                    +----------------------+
                    |  Microphone Input    |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | MFCC Extraction      |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Speaker Recognition  |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Speaker Prediction   |
                    +----------------------+
```

---

## Repository Structure

```text
AI_for_dementia_patients/

├── face/
│   ├── .gitignore
│   ├── README.md
│   ├── camera_recognition.py
│   ├── caregiver_setup.py
│   ├── evaluate_model.py
│   ├── fsl_model.py
│   ├── new_person_handler.py
│   ├── voice_output.py
│   └── requirements.txt
│
├── voice_recognition_system/
│   ├── audio_samples/
│   ├── dataset/
│   ├── embeddings/
│   ├── saved_embeddings/
│   ├── main.py
│   ├── record_test.py
│   └── test_voice.wav
│
├── streamlit_app.py
├── telegram_notifier.py
├── README.md
├── LICENSE
└── .gitignore
```

---

## Technology Stack

### Programming Languages

* Python

### Machine Learning

* PyTorch
* FaceNet
* CNN
* Few-Shot Learning
* Transfer Learning
* Incremental Learning

### Computer Vision

* OpenCV
* Face Detection
* Face Embeddings

### Audio Processing

* MFCC
* Speaker Embeddings
* Voice Feature Extraction

### Explainable AI

* GradCAM

### Deployment & Interface

* Streamlit

### Notifications

* Telegram Bot API

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/Harsha2oo5/AI_for_dementia_patients.git

cd AI_for_dementia_patients
```

### Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r face/requirements.txt
```

---

## Running the Face Recognition Module

Register known individuals:

```bash
python face/caregiver_setup.py
```

Start real-time recognition:

```bash
python face/camera_recognition.py
```

Features:

* Real-time face detection
* Relative recognition
* Context-aware memory prompts
* Unknown-person handling

---

## Running the Voice Recognition Module

```bash
cd voice_recognition_system

python main.py
```

Features:

* Speaker identification
* Voice embedding generation
* Voice-based recognition
* Incremental speaker enrollment

---

## Running the Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard provides:

* Recognition results
* User interaction interface
* Cognitive assistance outputs
* Real-time monitoring

---

## Machine Learning Approach

### Transfer Learning using FaceNet

The system leverages FaceNet (InceptionResNet-V1), pretrained on large-scale facial datasets.

Advantages:

* Strong facial representations
* High accuracy with limited data
* No extensive retraining required

Each facial image is converted into a 512-dimensional embedding vector.

---

### Few-Shot Learning

Traditional facial recognition systems require hundreds of images per individual.

This project uses Few-Shot Learning to:

* Learn new identities from only a few examples
* Adapt quickly to new relatives
* Reduce data collection requirements

For every registered individual:

1. Multiple facial embeddings are generated.
2. Embeddings are averaged to create a prototype representation.
3. Recognition is performed using cosine similarity.

---

### Speaker Recognition

Voice samples are processed using:

* MFCC feature extraction
* Speaker embeddings
* Similarity matching

This allows the system to identify individuals through voice in addition to facial information.

---

## Open-Set Recognition Workflow

The system handles three distinct scenarios.

### Scenario 1 — Unknown Person

A completely unfamiliar individual appears.

Actions:

* Store sample data
* Alert caregiver
* Avoid uncertain predictions

---

### Scenario 2 — Uncertain Match

The system finds a partial match but confidence is insufficient.

Actions:

* Mark prediction as uncertain
* Request caregiver verification
* Prevent incorrect memory retrieval

---

### Scenario 3 — New Person Registration

A caregiver wishes to register a new individual.

Actions:

* Capture samples
* Generate embeddings
* Create prototype representation
* Enable future recognition

---

## Explainable AI

To improve trust and transparency, the system incorporates Explainable AI techniques.

Features:

* GradCAM visualizations
* Confidence-aware predictions
* Interpretable model outputs

This helps caregivers understand how decisions are made by the AI system.

---

## Research Motivation

More than 55 million people worldwide live with dementia, often facing challenges in recognizing loved ones and recalling important personal information.

This project explores how Artificial Intelligence, Computer Vision, and Machine Learning can be combined to provide meaningful cognitive support and improve independence for dementia patients.

The long-term vision is to develop intelligent assistive technologies capable of acting as memory companions for individuals experiencing cognitive decline.

---

## Future Improvements

* Mobile application deployment
* Cloud synchronization
* Retrieval-Augmented Memory Systems (RAG)
* LLM-powered conversational assistance
* Multilingual voice support
* Wearable device integration
* Smart home connectivity
* Healthcare professional dashboards
* Real-time caregiver monitoring

---

## License

This project is licensed under the MIT License.

---

## Author

**K Sai Sri Harsha** 
**K M Skanda**

B.E. Electronics and Communication Engineering
B.M.S College of Engineering, Bengaluru

GitHub: https://github.com/Harsha2oo5

---
