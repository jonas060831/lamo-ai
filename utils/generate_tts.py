import base64
import uuid
import os
from TTS.api import TTS


# this model allow you to use a different voice
tts = TTS(model_name="tts_models/en/vctk/vits")

def generate_tts(text):
    filename = f"{uuid.uuid4()}.wav"

    tts.tts_to_file(text=text, speaker="p251", file_path=filename)

    with open(filename, "rb") as f:
        audio_bytes = f.read()

    os.remove(filename)

    return base64.b64encode(audio_bytes).decode("utf-8")
