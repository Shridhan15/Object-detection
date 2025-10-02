# Here we combines OCR, NLP, and object detection to analyze an image
from ocr_module import extract_text       # Extracts text from the image
from nlp_module import analyze_text       # Extracts entities from the text
from object_detection import detect_objects  # Detects objects in the image

def analyze_image(image_path):
    """
    Analyze an image and return a report:
     Extracted text (if any)
     Named entities found in the text (like dates, organizations, places)
     Objects detected in the image (like person, car, dog)

    Steps explained in simple terms:
     First, use OCR to read any text in the image.
     If text is found, use NLP to identify important pieces of information (entities).
     Then, use object detection to find common objects in the image.
     Combine all results into a single report.
    """
    report = {}

    #  OCR to extract text from the image
    text = extract_text(image_path)
    if text:
        report["Text Found"] = text  # Save the extracted text in the report

        #  NLP to extract named entities from text
        entities = analyze_text(text)
        if entities:
            report["Entities"] = entities  # Save entities in the report

    #  Object Detection to detect objects in the image
    objects = detect_objects(image_path)
    if objects:
        report["Objects Detected"] = objects  # Save object labels in the report

    # Return the report
    return report
