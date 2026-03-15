import uuid
import re
import os
import base64
from TTS.api import TTS
from utils.is_coding_question import replace_code_blocks
tts = TTS(model_name="tts_models/en/vctk/vits")




def generate_tts(text):

    # Replace URLs
    cleaned_text = re.sub(r'https?://[^\s]+', 'link provided for you.', text)

    # Replace code blocks
    cleaned_text = replace_code_blocks(cleaned_text)

    filename = f"{uuid.uuid4()}.wav"

    tts.tts_to_file(text=cleaned_text, speaker="p251", file_path=filename)

    with open(filename, "rb") as f:
        audio_bytes = f.read()

    os.remove(filename)

    return base64.b64encode(audio_bytes).decode("utf-8")
