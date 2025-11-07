# Here we combine OCR, NLP, object detection, and image captioning to analyze an image
from ocr_module import extract_text, extract_handwritten_text       # Extracts printed and handwritten text from the image
from nlp_module import analyze_text                                 # Extracts entities from the text
from object_detection import detect_objects                         # Detects objects in the image
from caption_module import generate_caption                         # Generates caption for the image


def analyze_image(image_path):
    """
    Analyze an image and return a report:
     - Extracted printed text (if any)
     - Extracted handwritten text (if any)
     - Named entities found in the printed text (like dates, organizations, places)
     - Objects detected in the image (like person, car, dog)
     - Caption describing the overall image content

    Steps explained in simple terms:
     1. Use OCR to read printed text in the image.
     2. If printed text is found, analyze it with NLP to extract entities.
     3. Separately, check for handwritten text using EasyOCR.
     4. Detect any common objects using the YOLOv8 model.
     5. Generate a simple natural caption describing the image.
     6. Combine all results into a single report dictionary.
    """

    report = {}

    # -------------------------------------------------------------------------
    # 🧾 Step 1: OCR to extract printed text from the image
    # -------------------------------------------------------------------------
    text = extract_text(image_path)
    if text:
        report["Text Found"] = text  # Save the extracted printed text

        # NLP to extract named entities (from printed text)
        entities = analyze_text(text)
        if entities:
            report["Entities"] = entities  # Save entities in the report

    # -------------------------------------------------------------------------
    # ✋ Step 2: OCR to extract handwritten text
    # -------------------------------------------------------------------------
    handwritten_text = extract_handwritten_text(image_path)
    if handwritten_text:
        report["Handwritten Text Found"] = handwritten_text  # Save handwritten text separately

    # -------------------------------------------------------------------------
    # 🎯 Step 3: Object Detection to detect objects in the image
    # -------------------------------------------------------------------------
    objects = detect_objects(image_path)
    if objects:
        report["Objects Detected"] = objects  # Save detected objects in the report

    # -------------------------------------------------------------------------
    # 🖼️ Step 4: Generate Caption using BLIP model
    # -------------------------------------------------------------------------
    caption = generate_caption(image_path)
    if caption:
        report["Image Caption"] = caption  # Save caption in the report

    # -------------------------------------------------------------------------
    # ✅ Step 5: Return the final combined analysis report
    # -------------------------------------------------------------------------
    return report
