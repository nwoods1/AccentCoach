import sys
import json
import os
from openai import OpenAI

import re
from openai import OpenAI
from g2p_en import G2p
import difflib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def transcribe_audio(audio_path):
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=open(audio_path, "rb"),
        language="en"  # Force language to English
    )
    return response.text


def analyze_audio(audio_path):
    transcription = transcribe_audio(audio_path)
    result = {
        "transcript": transcription
    }
    print(json.dumps(result))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <audio_path>")
        sys.exit(1)

    audio_path = sys.argv[1]
    analyze_audio(audio_path)
