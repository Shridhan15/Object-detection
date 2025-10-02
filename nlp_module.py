import spacy

# spaCy is a Python library for NLP.
# It helpsunderstand text by:
# Breaking text into words or tokens
# Identifying important entities like people, organizations, places, dates, and numbers
# Tagging parts of speech (noun, verb, etc.) and understanding word relationships

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

def analyze_text(text):
    """
    Function analyzes text using spaCy NLP and extracts Named Entities.

     The function looks through the text and finds important information
      like dates, organizations, places, numbers, etc. and then groups them by type and returns them in a dictionary.

    Example output:
    {
        "QUANTITY": ["4", "423-612-2102"],   
        # Things like amounts or measurable values 
        # (can be phone numbers).

        "DATE": ["May 2024", "June 2025 - August 2025"],  
        # Dates, time ranges, or periods.

        "ORG": ["Microsoft Office", "ABC Company"],  
        # Names of companies, organizations, or institutions.

        "CARDINAL": ["3.5"],  
        # Plain numbers without a unit, like count, or ranking.

        "GPE": ["Anytown", "Anytown USA"]  
        # Geographic (cities, states, countries).
    }

    Returns a dictionary where each key is the type of information (DATE, ORG, etc.)
              and the value is a list of the actual text found.
    """
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, []).append(ent.text)
    return entities
