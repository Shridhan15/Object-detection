import cv2
import pytesseract
from ultralytics import YOLO
import spacy

# Load YOLOv8n (small pretrained model)
model = YOLO("yolov8n.pt")

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Path to tesseract (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text(image_path):
    """Extract text from image using pytesseract"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()

def detect_objects(image_path):
    """Detect objects in image using YOLOv8"""
    results = model(image_path)
    objects = []
    for r in results[0].boxes.cls:
        label = results[0].names[int(r)]
        if label not in objects:
            objects.append(label)
    return objects

def analyze_text_with_nlp(text):
    """Extract entities from OCR text using spaCy"""
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    return entities

def analyze_image(image_path):
    """Return a friendly report of text + objects + entities"""
    report = {}

    # OCR
    text = extract_text(image_path)
    if text:
        report["Text Found"] = text

        # Run NLP only if text exists
        entities = analyze_text_with_nlp(text)
        if entities:
            report["Entities"] = entities

    # Object Detection
    objects = detect_objects(image_path)
    if objects:
        report["Objects Detected"] = objects

    return report
