from langdetect import detect, detect_langs
import re
import numpy as np
from sentence_transformers import SentenceTransformer

def clean_text(text):
    
    text_lower = text.lower()
    url_pattern = r"http\S+"
    no_urls = re.sub(url_pattern, "", text_lower)
    allowed = r"[^a-zA-Z\u0B80-\u0BFF ]"
    clean = re.sub(allowed, "", no_urls)
    return clean.strip()

def char_ngrams(text, n=3):

    ngramlist = []
    textlength = len(text)
    maxindx = textlength-n+1
    i = 0
    for i in range(maxindx):
        ngram = text[i:i+n]
        ngramlist.append(ngram)
    return ngramlist

model =SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(ngramslist):

    embeddingdict = {}
    for ng in ngramslist:
        vector = model.encode(ng)
        embeddingdict[ng] = vector

    return embeddingdict

def sentencevector(embeddings):
    if not embeddings:
        return np.zeros(999)
    
    vector_list = list(embeddings.values())
    sentence_vector = np.mean(vector_list, axis=0)
    return sentence_vector
def detect_language(text, top_k = 5):
    results = []
    try:
        detected_languages = detect_langs(text)
        count = 0
        for detected in detected_languages:
            if count >= top_k:
                break
            language=detected.lang
            confidence=detected.prob
            results.append((language, confidence))
            count += 1

    except Exception:
        results = [("unknown", 0.0)]

    return results



def run_pipeline():
    
    print("\nNLP LANGUAGE AUTOMATIC DETECTION\n")
    text = input("Enter text to detect:\n> ")
    cleaned = clean_text(text)
    
    print("\nCleaned Text:")
    print(cleaned)
    
    ngrams = char_ngrams(cleaned, n=3)
    print("\nCharacter n-grams (first 15):")
    print(ngrams[:15])
    
    embeddings = generate_embeddings(ngrams)
    print("\nSample N-gram Embeddings:")
    for k, v in embeddings.items():
        print(k, "→", v)
    
    sentence_vector = sentencevector(embeddings)
    print("\nSentence Vector:")
    print(sentence_vector)
    
    predictions = detect_language(cleaned, top_k=5)
    print("\nLanguage Predicted")
    for lang, conf in predictions:
        print(f"{lang} → Confidence: {conf:.4f}")
   
    best_lang,best_conf = predictions[0]
    print("\nFINAL OUTPUT")
    print("Detected Language:",best_lang)
    print("Confidence Score:",best_conf)

if __name__ == "__main__":
    run_pipeline()
