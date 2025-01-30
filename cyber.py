from transformers import pipeline
from googletrans import Translator

# Load the pre-trained model.
classifier = pipeline("text-classification", model="s-nlp/roberta_toxicity_classifier")

# Initialize the Google Translate translator
translator = Translator()

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '' if t.startswith('@') and len(t) > 1 else t
        t = '' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def detect_cyberbullying(text):
    try:
        # Detect the language of the input text
        ptext=preprocess(text)
        detected_lang = translator.detect(ptext).lang

        # Translate to English if not already English
        if detected_lang != 'en':
            translated = translator.translate(ptext, dest='en').text
        else:
            translated = ptext

        # Perform cyberbullying detection on the translated text
        result = classifier(translated)[0]

        label = result['label']
        
        return label # Return the label with the highest score
    except Exception as e:
        print(f"Error during classification: {e}")
        return "UNKNOWN"
