import whisper
import os

model = whisper.load_model("base")

def generate_stt(audio_file):
    temp_path = "temp_audio.webm" #save only the webm for now
    audio_file.save(temp_path)

    #process the audio file
    result = model.transcribe(temp_path,language='en',fp16=False)

    #Cleanup
    os.remove(temp_path)

    return result['text']