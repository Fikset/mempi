# Voice Assistant with Long Term Memory using Azure, OpenAI, and Elasticsearch

This project implements a conversational voice assistant on a Raspberry Pi (Legacy, 64-bit) with the following features:

- Speech recognition and transcription (via OpenAI Whisper)
- Text-to-speech (via Azure Cognitive Services)
- ChatGPT-based conversation (via OpenAI GPT models)
- Long-term memory storage and retrieval (via Elasticsearch)

For hardware details and connections (microphone, speaker, and button), follow the instructions in the [Instructables article](https://www.instructables.com/Customizes-a-ChatGPT-Assistant-Using-a-RaspberryPi/).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Hardware Requirements](#hardware-requirements)
- [Setup](#setup)
  - [1. Environment Variables](#1-environment-variables)
  - [2. Install Dependencies](#2-install-dependencies)
  - [3. Configure Elasticsearch](#3-configure-elasticsearch)
  - [4. Run the Application](#4-run-the-application)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- A Raspberry Pi running **Raspberry Pi OS (Legacy, 64-bit)**.  
- A working internet connection.
- An active [OpenAI API key](https://platform.openai.com/) for GPT and Whisper usage.
- An [Azure Cognitive Services Speech resource](https://azure.microsoft.com/en-us/services/cognitive-services/text-to-speech/) with a valid subscription key and region.
- An Elasticsearch endpoint (local or remote) if you want to use long-term memory features.

---

## Hardware Requirements

1. **USB Microphone** or microphone connected via audio interface (e.g., USB sound card).  
2. **Speaker** connected to the Raspberry Pi audio out or via USB.  
3. **Button** (optional but included in the code), connected to a GPIO pin for resetting the conversation.  
4. **Power supply**, cables, and other basic Raspberry Pi accessories (microSD card, etc.).

For detailed wiring instructions, refer to the [Instructables article](https://www.instructables.com/Customizes-a-ChatGPT-Assistant-Using-a-RaspberryPi/).

---

## Setup

### 1. Environment Variables

Set the following environment variables on your Raspberry Pi. You can use a file like `~/.bashrc` or `~/.profile` to store them:

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export SPEECH_KEY="YOUR_AZURE_SPEECH_KEY"
export SPEECH_REGION="YOUR_AZURE_SPEECH_REGION"
export ELASTIC_ENDPOINT="YOUR_ELASTICSEARCH_ENDPOINT"
export ELASTIC_KEY="YOUR_ELASTICSEARCH_API_KEY"
```

Replace `YOUR_OPENAI_API_KEY`, `YOUR_AZURE_SPEECH_KEY`, etc., with your own credentials.

### 2. Install Dependencies

Make sure your Raspberry Pi is up to date:
```bash
sudo apt-get update
sudo apt-get upgrade
```

Install Python 3, pip, and other packages if not already available:
```bash
sudo apt-get install python3 python3-pip python3-dev
```

Clone this repository or place the code on your Pi, then install the required Python libraries:
```bash
pip3 install -r requirements.txt
```

> **Note**: The code uses libraries like `azure.cognitiveservices.speech`, `openai`, `speech_recognition`, `sounddevice`, `sentence-transformers`, and more. Check `requirements.txt` for the exact list.

### 3. Configure Elasticsearch

If you haven’t done so already, install or configure your Elasticsearch instance. You can run Elasticsearch locally on your Pi or connect to a remote server. Ensure the endpoint and API key are set via the environment variables above.

### 4. Run the Application

With everything configured, navigate to the project folder and start the application:
```bash
python3 main.py
```

The assistant will greet you with a spoken prompt (“How may I help?”). When you speak, it should transcribe your speech, pass it to ChatGPT, and then speak back the response. It will also store conversation data in Elasticsearch for long-term memory.

---

## Usage

1. **Speak a question or command** once you hear the prompt.
2. The assistant will **transcribe** your speech using OpenAI Whisper.
3. It will **retrieve** any relevant context from Elasticsearch (long-term memory).
4. It will **generate a response** using OpenAI’s ChatGPT endpoint.
5. Azure Cognitive Services will **speak** the assistant’s reply.
6. If you press the **button** (wired to `board.D16` in the code), the program will restart.

---

## Troubleshooting

- **Microphone not detected**: Check `arecord -l` or `lsusb` to ensure your microphone is recognized. Update `microphone` references if needed.
- **Azure Speech or OpenAI API errors**: Make sure your credentials are correct and you have an active subscription/usage quota.
- **Elasticsearch not reachable**: Verify your Elasticsearch endpoint and API key, and ensure your Pi can connect to it.
- **Button wiring**: Double-check GPIO pin references if the script doesn’t restart on button press.
