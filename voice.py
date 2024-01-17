import streamlit as st
from safetensors import torch
from transformers import pipeline
import torchaudio

@st.cache
def load_models():
    asr_model = pipeline(task="automatic-speech-recognition", model="NVIDIA/QuartzNet15x5Base-En")
    nlp_model = pipeline("text-generation", model="gpt2")
    return asr_model, nlp_model


def main():
    st.title("Doctor-Patient Voice Chatbot")
    st.write("This chatbot allows you to communicate with a doctor using your voice.")

    # Load the models
    asr_model, nlp_model = load_models()

    # Get the user's voice message
    st.write("Please upload your voice message.")
    audio_file = st.file_uploader("Audio file", type="mp3")

    # Transcribe the audio message
    if audio_file is not None:
        transcribed_text = transcribe_audio(audio_file, asr_model)
        st.write("Your voice message was transcribed as follows:")
        st.write(transcribed_text)

        # Generate a response
        chatbot_response = generate_response(transcribed_text, nlp_model)
        st.write("The chatbot responded with the following:")
        st.write(chatbot_response)

def transcribe_audio(audio_file, asr_model):
    waveform, sample_rate = torchaudio.load(audio_file)
