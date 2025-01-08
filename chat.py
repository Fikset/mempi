import logging
import os
import re
import sys
import threading
import time
from datetime import datetime, timedelta
from queue import Queue

import azure.cognitiveservices.speech as speechsdk
import board
import digitalio
import openai
import sounddevice
import speech_recognition as sr
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

logging.basicConfig(level="ERROR")

# ChatGPT Parameters
SYSTEM_ROLE = (
    "Your role is a voice assistant. Answers must be plain text so they could be read aloud. "
    "No formulas, no code, markup. You do remember past conversations. "
    "You will receive context from your long term memory sometimes."
)

CHATGPT_MODEL = "gpt-4o"
WHISPER_MODEL = "whisper-1"

# Azure Parameters
AZURE_SPEECH_VOICE = "en-US-AvaMultilingualNeural"
DEVICE_ID = None

# Speech Recognition Parameters
ENERGY_THRESHOLD = 100      # Energy level threshold for the mic
PHRASE_TIMEOUT = 2.5        # Time gap to separate phrases
NEW_CHAT_TIMEOUT = 7.0
RECORD_TIMEOUT = NEW_CHAT_TIMEOUT

# Environment Variables
openai.api_key = os.environ.get("OPENAI_API_KEY")
speech_key = os.environ.get("SPEECH_KEY")
service_region = os.environ.get("SPEECH_REGION")
elastic_endpoint = os.environ.get("ELASTIC_ENDPOINT")
elastic_key = os.environ.get("ELASTIC_KEY")

if openai.api_key is None or speech_key is None or service_region is None:
    print("Please set the OPENAI_API_KEY, SPEECH_KEY, and SPEECH_REGION environment variables first.")
    sys.exit(1)

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_synthesis_voice_name = AZURE_SPEECH_VOICE


def send_chat(conversation_history):
    """Send conversation history to ChatGPT and return its reply with updated conversation history."""
    completion = openai.chat.completions.create(
        model=CHATGPT_MODEL,
        messages=conversation_history,
    )
    assistant_reply = completion.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply, conversation_history


def make_summary(conversation_history, summaries: Queue):
    """
    Ask ChatGPT to create/merge a summary of the current conversation,
    and put the result into a queue for background usage.
    """
    SUMMARY_ROLE = "Your task is to merge the existing summary with the new conversation."
    summary_prompt = (
        "Reply with the summary of our conversation in format "
        "[{'fact': 'value', 'sentiment': 'value', 'importance': 'value'}, ]"
    )

    history = conversation_history.copy()
    history[0] = {"role": "system", "content": SUMMARY_ROLE}
    history.append({"role": "user", "content": summary_prompt})

    print(f"history: {history}")

    completion = openai.chat.completions.create(
        model=CHATGPT_MODEL,
        messages=history,
    )
    summary = completion.choices[0].message.content
    summaries.put(summary)
    print(f"Summary: {summary}")


def transcribe(wav_data):
    """
    Transcribe audio bytes using OpenAI's Whisper endpoint.
    Retry up to 3 times if the API returns an error.
    """
    print("Transcribing...")
    attempts = 0
    while attempts < 3:
        try:
            with open('temp.wav', 'wb+') as temp_file:
                temp_file.write(wav_data)
                temp_file.flush()
                temp_file.seek(0)
                result = openai.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=temp_file
                )
                return result.text.strip()
        except openai.APIError:
            time.sleep(3)
            attempts += 1
    return "I wasn't able to understand you. Please repeat that."


class LongTermMemory:
    """
    Manages saving and retrieving conversation data from Elasticsearch using vector embeddings.
    """
    def __init__(self):
        self.es = Elasticsearch(
            elastic_endpoint,
            api_key=elastic_key,
        )
        self.vectorization_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index_name = 'semantic_texts'

        if not self.es.indices.exists(index=self.index_name):
            self.create_index()

    def create_index(self):
        """
        Create or recreate the Elasticsearch index with a dense_vector mapping.
        """
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)

        mapping = {
            "mappings": {
                "properties": {
                    "role": {"type": "keyword"},
                    "content": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 384  # Dimensions match the chosen model
                    }
                }
            }
        }
        self.es.indices.create(index=self.index_name, body=mapping)

    def save_ltm(self, conversation_history):
        """
        Save user/assistant conversation to Elasticsearch with vector embeddings.
        Skips if there's no meaningful conversation yet.
        """
        if len(conversation_history) <= 1:
            logging.warning("No history to save")
            return
        messages = conversation_history[1:]  # skip the system message
        texts = [msg["content"] for msg in messages]
        embeddings = self.vectorization_model.encode(texts, convert_to_tensor=False)

        actions = []
        for msg, embedding in zip(messages, embeddings):
            actions.append({
                "_index": self.index_name,
                "_source": {
                    "role": msg["role"],
                    "content": msg["content"],
                    "embedding": embedding.tolist()
                }
            })

        helpers.bulk(self.es, actions)
        self.es.indices.refresh(index=self.index_name)

    def search_and_add_ltm(self, prompt, threshold=0.2):
        """
        Searches the long-term memory for context similar to `prompt`.
        Appends context to the end of the prompt for optional consideration.
        """
        query_embedding = self.vectorization_model.encode(prompt, convert_to_tensor=False)
        min_score = 2.0 - threshold

        response = self.es.search(
            index=self.index_name,
            body={
                "size": 5,
                "min_score": min_score,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": query_embedding.tolist()}
                        }
                    }
                }
            }
        )

        new_prompt = prompt
        for hit in response['hits']['hits']:
            score = hit['_score'] - 1.0  # Adjust the score to normal cosine similarity
            content = hit['_source']['content']
            role = hit['_source']['role']
            logging.info(f"Score: {score:.4f}, Role: {role}, Content: {content}")
            print(f"Score: {score:.4f}, Role: {role}, Content: {content}")
            new_prompt += f"""
            
            Above is context from your long term memory. Optional for consideration.
            Role: {role}, Content: {content}
            """
        logging.debug(new_prompt)
        return new_prompt


class Listener:
    """Handles microphone input and queues audio data for transcription."""
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = ENERGY_THRESHOLD
        self.recognizer.dynamic_energy_threshold = False
        self.recognizer.pause_threshold = 1
        self.last_sample = bytes()
        self.phrase_time = datetime.utcnow()
        self.phrase_timeout = PHRASE_TIMEOUT
        self.phrase_complete = False
        self.data_queue = Queue()

    def listen_once(self):
        """Listen for a single utterance and store raw audio data in a queue."""
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=RECORD_TIMEOUT)
            data = audio.get_raw_data()
            self.data_queue.put(data)
        except sr.WaitTimeoutError:
            pass

    def speech_waiting(self):
        """Check if there's queued audio data."""
        return not self.data_queue.empty()

    def get_speech(self):
        """Retrieve queued audio data if available."""
        if self.speech_waiting():
            return self.data_queue.get()
        return None

    def get_audio_data(self):
        """
        Retrieve all queued samples, reset if phrase timeout is exceeded,
        and combine them into a single AudioData object.
        """
        now = datetime.utcnow()

        if self.speech_waiting():
            self.phrase_complete = False
            if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
                self.last_sample = bytes()
                self.phrase_complete = True

            self.phrase_time = now

            while self.speech_waiting():
                data = self.get_speech()
                self.last_sample += data

            with sr.Microphone() as source:
                audio_data = sr.AudioData(
                    self.last_sample,
                    source.SAMPLE_RATE,
                    source.SAMPLE_WIDTH
                )
            return audio_data
        return None


class Chat:
    """
    Manages text-to-speech output using Azure, and also checks button states for conversation restarts.
    """
    def __init__(self, azure_speech_config):
        self._button = digitalio.DigitalInOut(board.D16)
        self._button.direction = digitalio.Direction.INPUT
        self._button.pull = digitalio.Pull.UP

        if DEVICE_ID is None:
            audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        else:
            audio_config = speechsdk.audio.AudioOutputConfig(device_name=DEVICE_ID)

        self._speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=azure_speech_config,
            audio_config=audio_config
        )

    def deinit(self):
        """Disconnect any event handlers from the speech synthesizer."""
        self._speech_synthesizer.synthesis_started.disconnect_all()
        self._speech_synthesizer.synthesis_completed.disconnect_all()

    def button_pressed(self):
        """Return True if the physical button is pressed."""
        return not self._button.value

    def speak(self, text):
        """Convert text to speech using Azure Cognitive Services."""
        result = self._speech_synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"Speech synthesized for text [{text}]")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation_details.error_details}")

    def stop_speaking(self):
        """Stop ongoing speech synthesis."""
        try:
            self._speech_synthesizer.stop_speaking_async()
        except Exception as e:
            print(e)


def main():
    listener = Listener()
    chat = Chat(speech_config)
    ltm = LongTermMemory()

    transcription = []
    conversation_history = [{"role": "system", "content": SYSTEM_ROLE}]

    chat.speak("How may I help?")
    listener.listen_once()

    def restart_conversation():
        """Monitor button press and restart the script if needed."""
        while True:
            if chat.button_pressed():
                print("Restarting conversation...")
                os.execl(sys.executable, sys.executable, *sys.argv)
            time.sleep(0.1)

    # Start a background thread to monitor the restart button
    threading.Thread(target=restart_conversation, name="restart_conversation", daemon=True).start()

    while True:
        try:
            if listener.speech_waiting():
                audio_data = listener.get_audio_data()
                if not audio_data:
                    time.sleep(0.2)
                    continue

                chat.speak("Recognising...")
                text = transcribe(audio_data.get_wav_data())
                print(f"Recognised: {text}")

                # Check if user said something beyond trivial filler
                if len(re.sub("[^a-zA-ZА-Яа-я]+", "", text)) > 1 and text.lower() != "you" and listener.phrase_complete:
                    chat.speak("Thinking...")
                    transcription.append(text)

                    # Attempt to add relevant LTM context
                    text_with_context = ltm.search_and_add_ltm(text)
                    conversation_history.append({"role": "user", "content": text_with_context})

                    print(f"Phrase Complete. Sent '{text}' to ChatGPT.")
                    chat_response, conversation_history = send_chat(conversation_history)
                    transcription.append(f"> {chat_response}")

                    chat.speak(chat_response)
                    listener.listen_once()
                else:
                    print("Updating long term memory on the background...")

                    chat.speak("Writing to the long term memory.")
                    ltm.save_ltm(conversation_history)
                    chat.speak("Done. Goodbye!")

                for line in transcription:
                    print(line)

                print("sleep .25")
                time.sleep(0.25)

            time.sleep(0.2)

        except KeyboardInterrupt:
            print("Exiting...")
            break


if __name__ == "__main__":
    main()
