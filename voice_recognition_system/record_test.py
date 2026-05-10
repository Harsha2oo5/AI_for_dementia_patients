import sounddevice as sd
from scipy.io.wavfile import write

# Recording settings
sample_rate = 16000
duration = 10  # seconds

print("Recording will start...")

# Record audio
audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype='float32'
)

sd.wait()

# Save recording
write("test_voice.wav", sample_rate, audio)

print("Recording completed.")
print("Saved as test_voice.wav")