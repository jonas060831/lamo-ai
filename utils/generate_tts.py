import base64
import uuid
import os
from TTS.api import TTS


# this model allow you to use a different voice
import re
import uuid
import os
import base64
from TTS.api import TTS

tts = TTS(model_name="tts_models/en/vctk/vits")

def generate_tts(text):
    # Replace URLs with "link provided. for you."
    cleaned_text = re.sub(r'https?://[^\s]+', 'link provided. for you.', text)

    filename = f"{uuid.uuid4()}.wav"

    tts.tts_to_file(text=cleaned_text, speaker="p251", file_path=filename)

    with open(filename, "rb") as f:
        audio_bytes = f.read()

    os.remove(filename)

    return base64.b64encode(audio_bytes).decode("utf-8")
