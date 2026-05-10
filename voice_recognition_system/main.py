from speechbrain.inference.speaker import EncoderClassifier
import soundfile as sf
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# =====================================================
# LOAD ECAPA MODEL
# =====================================================

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

# =====================================================
# GENERATE EMBEDDING
# =====================================================

def get_embedding(audio_path):

    signal, fs = sf.read(audio_path)

    signal = torch.tensor(signal).float().unsqueeze(0)

    embedding = classifier.encode_batch(signal)

    return embedding.squeeze(0).detach().numpy()

# =====================================================
# SAVE EMBEDDING
# =====================================================

def save_embedding(person_name, embedding):

    save_path = f"saved_embeddings/{person_name}.npy"

    np.save(save_path, embedding)

    print(f"\nNew patient enrolled: {person_name}")

# =====================================================
# LOAD DATABASE
# =====================================================

def load_database():

    database = {}

    for file in os.listdir("saved_embeddings"):

        if file.endswith(".npy"):

            person_name = file.replace(".npy", "")

            embedding = np.load(
                f"saved_embeddings/{file}"
            )

            database[person_name] = embedding

    return database

# =====================================================
# LOAD DATABASE
# =====================================================

database = load_database()

# =====================================================
# LIVE RECORDED AUDIO
# =====================================================

test_audio_path = "test_voice.wav"

test_embedding = get_embedding(test_audio_path)

# =====================================================
# SPEAKER IDENTIFICATION
# =====================================================

best_match = None
best_score = -1

print("\n========== SIMILARITY SCORES ==========\n")

for person_name, stored_embedding in database.items():

    similarity = cosine_similarity(
        test_embedding,
        stored_embedding
    )[0][0]

    print(f"{person_name}: {similarity:.4f}")

    if similarity > best_score:
        best_score = similarity
        best_match = person_name

# =====================================================
# FINAL DECISION
# =====================================================

THRESHOLD = 0.60

print("\n========== FINAL RESULT ==========\n")

if best_score > THRESHOLD:

    print(f"Recognized Patient : {best_match}")
    print(f"Confidence Score   : {best_score:.4f}")

else:

    print("Unknown Patient")

    new_patient_name = input(
        "\nEnter new patient name: "
    )

    save_embedding(
        new_patient_name,
        test_embedding
    )

print("\n==================================")