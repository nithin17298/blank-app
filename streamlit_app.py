import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
ROBERTA_SUPPORTED_LANGUAGES = ('ar', 'en', 'fr', 'de', 'hi', 'it', 'es', 'pt')

model = AutoModelForSequenceClassification.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

#/ save the model locally
model.save_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def predict_sentiment(text: str) -> str:
    processed_text = preprocess(text)
    encoded_input = tokenizer(processed_text, return_tensors='pt')
    output = model(**encoded_input)
    index_of_sentiment = output.logits.argmax().item()
    sentiment = config.id2label[index_of_sentiment]
    return sentiment

st.title("Audio to Text Converter")

input_option = st.radio("Select input type:", ("Text", "Audio"))

if input_option == "Text":
        text_input = st.text_area("Enter text:")
        if st.button("Transcribe"):
            if text_input:
                st.write("Transcribed Text:")
                st.write(text_input)
                st.write("Sentiment:") 
                st.write(predict_sentiment(text_input)) # Display the entered text directly
            else:
                st.warning("Please enter some text.")


elif input_option == "Audio":

 uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

 if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file, format='audio/wav')

    # Process audio
    try:
        # Create a temporary file path for the uploaded audio
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Initialize recognizer
        r = sr.Recognizer()

        # Load audio file
        with sr.AudioFile("temp_audio.wav") as source:
            audio_data = r.record(source)

        # Recognize speech
        text = r.recognize_google(audio_data)
        
        st.success("Transcription successful!")
        st.write("Transcribed Text:")
        st.write(text)
        st.write("Sentiment:") 
        st.write(predict_sentiment(text)) # Display the entered text directly


    except sr.UnknownValueError:
        st.error("Could not understand audio")
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")




