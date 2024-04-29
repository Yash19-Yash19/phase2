import streamlit as st
import json
import torch
from model import NeuralNet
from nltk_x1 import tokenize, bag_of_words
import numpy as np
import speech_recognition as sr
from gtts import gTTS
import os
import winsound

# Load intents and pre-trained model
with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Function to get text input


def get_text_input():
    return st.text_input("You:")

# Function to get voice input


def get_voice_input():
    audio_bytes = None
    with st.spinner("Listening..."):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            audio = recognizer.listen(source)
        try:
            input_text = recognizer.recognize_google(audio)
            st.write("You:", input_text)
            return input_text
        except sr.UnknownValueError:
            return None

# Function to convert text to speech


def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    winsound.PlaySound("output.mp3", winsound.SND_FILENAME)

# Function to get response from the chatbot


def get_response(input_text):
    sentence = tokenize(input_text)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.80:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                # Selecting the first response
                response = intent['responses'][0]
                return response
    else:
        return "I do not understand..."

# Main function to run the Streamlit app


def main():
    st.title("Chatbot")
    is_recording = False

    option = st.radio("Select Input Option:", ("Text", "Voice"))

    if option == "Text":
        input_text = get_text_input()
        if input_text and st.button("Submit"):
            response = get_response(input_text)
            st.write("Bot:", response)
            text_to_speech(response)
            # Display audio output
            audio_file = gTTS(response, lang='en', slow=False)
            audio_file.save("output.mp3")
            audio_bytes = open("output.mp3", "rb").read()
            st.audio(audio_bytes, format='audio/mp3', start_time=0)
            if st.button("Ask Another Question"):
                main()  # Recursive call to start over
    else:
        input_text = get_voice_input()
        if input_text:
            response = get_response(input_text)
            st.write("Bot:", response)
            text_to_speech(response)
            # Display audio output
            audio_file = gTTS(response, lang='en', slow=False)
            audio_file.save("output.mp3")
            audio_bytes = open("output.mp3", "rb").read()
            st.audio(audio_bytes, format='audio/mp3', start_time=0)
            if st.button("Ask Another Question"):
                main()  # Recursive call to start over


# Run the app
if __name__ == "__main__":
    main()
