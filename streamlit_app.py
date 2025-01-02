import streamlit as st
import speech_recognition as sr
import re
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from translate import translate_text
#from cyber import detect_cyberbullying
#from googletrans import Translator
#from transformers import logging


MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
ROBERTA_SUPPORTED_LANGUAGES = ('ar', 'en', 'fr', 'de', 'hi', 'it', 'es', 'pt')

model = AutoModelForSequenceClassification.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

#/ save the model locally
model.save_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)


#cyberbullying_classifier = "s-nlp/roberta_toxicity_classifier"

#model = AutoModelForSequenceClassification.from_pretrained(cyberbullying_classifier)
#tokenizer = AutoTokenizer.from_pretrained(cyberbullying_classifier)
#config = AutoConfig.from_pretrained(cyberbullying_classifier)

#translator = Translator()  # For language translation


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '' if t.startswith('@') and len(t) > 1 else t
        t = '' if t.startswith('http') else t
        new_text.append(t)
    #clean_text= " ".join(new_text)
    #clean_text = re.sub(r'[^\w\s]', '', clean_text)
    #clean_text = re.sub(r'[_\W]+', ' ', clean_text) 
    #clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    #if not clean_text or clean_text.isspace():
        #clean_text = "[EMPTY]" 
    return " ".join(new_text)

def predict_sentiment(text: str) -> str:
    processed_text = preprocess(text)
    encoded_input = tokenizer(processed_text, return_tensors='pt')
    output = model(**encoded_input)
    index_of_sentiment = output.logits.argmax().item()
    sentiment = config.id2label[index_of_sentiment]
    return sentiment

#def detect_cyberbullying(text):
#   """Detects cyberbullying, translating to English if needed."""
#   try:
#       detected_lang = translator.detect(text).lang
#       if detected_lang != 'en':
#            translated = translator.translate(text, dest='en').text
#       else:
#           translated = text
#
 #       result = cyberbullying_classifier(translated)[0]
 # #      label = result['label']
  #      if label == "toxic":
    #        label = "cyberbullying"
     #   return label
    #except Exception as e:
     #   print(f"Error during classification: {e}")
      #  return "UNKNOWN"


st.title("Cyber Bullying Detection")

input_option = st.radio("Select input type:", ("Text", "Audio"))

if input_option == "Text":
        text_input = st.text_area("Enter text:")
        st.markdown("*Don't enter only symbols.*")
        if st.button("Transcribe"):
            if text_input:
                st.write("Transcribed Text:")
                text_input=translate_text(text_input)
                #st.write(preprocess(text_input)) 
                label=predict_sentiment(text_input)
                st.write("Detection:") 
                if label=="negative":
                    st.write("toxic")
                elif label=="positive":
                    st.write("not_toxic")
                else:
                    st.write("neutral")
                st.write("Sentiment:") 
                st.write(label)
                
                #cyberbullying_label= detect_cyberbullying(text_input)
                #st.write(f"Cyberbullying Detection: {cyberbullying_label}")
# Display the entered text directly
            else:
                st.warning("Please enter some text.")


elif input_option == "Audio":

 uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

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
        label=predict_sentiment(text)
        st.write("Detection:") 
        st.write(detect_cyberbullying(text))
        st.write("Sentiment:") 
        st.write(label)
        
                 # Display the entered text directly


    except sr.UnknownValueError:
        st.error("Could not understand audio")
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")




