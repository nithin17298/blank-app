from transformers import pipeline
from googletrans import Translator

# Load the pre-trained model.
classifier = pipeline("text-classification", model="s-nlp/roberta_toxicity_classifier")

# Initialize the Google Translate translator
translator = Translator()

def detect_cyberbullying(text):
    try:
        # Detect the language of the input text
        detected_lang = translator.detect(text).lang

        # Translate to English if not already English
        if detected_lang != 'en':
            translated = translator.translate(text, dest='en').text
        else:
            translated = text

        # Perform cyberbullying detection on the translated text
        result = classifier(translated)[0]

        label = result['label']
        if label == "toxic":
            label = "cyberbullying"

        return label # Return the label with the highest score
    except Exception as e:
        print(f"Error during classification: {e}")
        return "UNKNOWN"
