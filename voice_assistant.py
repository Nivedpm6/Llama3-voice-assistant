import ollama
import os
import sounddevice as sd
import wavio
import assemblyai as aai
import requests
import pygame
import tempfile
import numpy as np
import threading
# Initialize the pygame mixer
pygame.mixer.init()


def record_audio(filename, samplerate=44100):
    recording = []
    recording_event = threading.Event()

    def callback(indata, frames, time, status):
        recording.append(indata.copy())
        if recording_event.is_set():
            raise sd.CallbackAbort

    print("Recording... Press Enter to stop.")
    stream = sd.InputStream(samplerate=samplerate, channels=2, callback=callback)

    with stream:
        input()  # Wait for the user to press Enter
        recording_event.set()

    # Combine recorded chunks into a single array
    audio = np.concatenate(recording, axis=0)

    # Save the recording to a file
    wavio.write(filename, audio, samplerate, sampwidth=2)
    #print(f"Audio saved as {filename}")

# Example usage


def chat_with_model():
    print("Connected to Llama3")
    print("Type 'exit' to end the chat.")

    conversation_history = []
    
    while True:
        record_audio("recorded_audio.wav")

        # Set your AssemblyAI API key
        aai.settings.api_key = "aai_api"

        # Initialize the transcriber
        transcriber = aai.Transcriber()

        # Transcribe the recorded audio
        transcript = transcriber.transcribe("recorded_audio.wav")
        message = transcript.text
        print("Nived : ", message)
        if message.lower() == 'exit.':
            break

        # Add the user's message to the conversation history
        conversation_history.append({'role': 'user', 'content': message})

        try:
            response_stream = ollama.chat(
                model="llama3",
                messages=conversation_history,
                stream=True
            )

            response_content = ""
            print("Assistant : ")
            for response in response_stream:
                if 'message' in response and 'content' in response['message']:
                    content = response['message']['content']
                    response_content += content
                    print(content, end='')  # Print without adding a newline
                else:
                    print("Unexpected response structure:", response)

            print()  # Newline after the complete response

            CHUNK_SIZE = 1024
            url = "https://api.elevenlabs.io/v1/text-to-speech/j9RedbMRSNQ74PyikQwD"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": "xi_api"
            }
            data = {
                "text": response_content,
                "model_id": "eleven_monolingual_v1",
                "stream": True,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }

            # Create a temporary file for the MP3 data
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_file_name = temp_file.name
                response = requests.post(url, json=data, headers=headers)
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        temp_file.write(chunk)

            #print("Generated speech saved temporarily as", temp_file_name)

            # Load the MP3 file
            pygame.mixer.music.load(temp_file_name)

            # Play the MP3 file
            pygame.mixer.music.play()

            # Keep the program running until the music stops
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            # Delete the temporary file after playback
            os.remove(temp_file_name)
            print("Temporary file deleted after playback")

            # Add the model's response to the conversation history
            conversation_history.append({'role': 'assistant', 'content': response_content})

        except Exception as e:
            print("")

# Start the chatbot
chat_with_model()
