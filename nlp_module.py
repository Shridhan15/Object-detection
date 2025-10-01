# nlp_module.py
import spacy

nlp = spacy.load("en_core_web_sm")

def process_text(text):
    doc = nlp(text)
    
    entities = {}
    for ent in doc.ents:
        entities[ent.label_] = entities.get(ent.label_, []) + [ent.text]
    
    # Optional: return cleaned text too
    cleaned_text = " ".join([token.text for token in doc if not token.is_punct])
    
    return cleaned_text, entities
