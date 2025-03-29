import os
from groq import Groq

class TextToSpeech:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))  # Initialize Groq client

    def speak_male(self, text: str, filename: str = "male_speech.wav"):
        """Convert text to speech using Groq API and save it as an audio file."""
        model = "playai-tts"
        voice = "Fritz-PlayAI"
        response_format = "wav"

        response = self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format
        )

        response.write_to_file(filename)  # Save the audio file
        print(f"Audio saved as {filename}")

    def speak_female(self, text: str, filename: str = "female_speech.wav"):
        model = "playai-tts"
        voice = "Arista-PlayAI"
        response_format = "wav"

        response = self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format
        )

        response.write_to_file(filename)  # Save the audio file
        print(f"Audio saved as {filename}")
