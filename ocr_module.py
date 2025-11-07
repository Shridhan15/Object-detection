import cv2
import pytesseract
from PIL import Image
import easyocr   # EasyOCR for better handwritten text detection

# Tesseract is an open-source OCR engine
# It can read text from images and return it.
# Tesseract must be installed on your system for this to work.

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# EasyOCR is a lightweight deep learning OCR library
# It can detect both printed and handwritten text accurately.
# We initialize it once globally (to avoid reloading on each Flask request)
reader = easyocr.Reader(['en'], gpu=True)



def extract_text(image_path):
    """
    Extract text from an image using pytesseract OCR.

    Steps in simple terms:
    1. Load the image from the given file path.
    2. Convert the image to grayscale (OCR works better on simpler images).
    3. Send the image to Tesseract OCR to read the text.
    4. Return the text as a Python string.

    Example:
        Image: [Picture with "Invoice #12345"]
        Output: "Invoice #12345"
    """
    # Step 1: Read the image
    img = cv2.imread(image_path)

    # Step 2: Convert the image to grayscale (black and white shades only) 
    # so that Tesseract OCR can focus on the text shapes.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply Tesseract OCR to detect text
    text = pytesseract.image_to_string(gray)

    # Step 4: Return cleaned text
    return text.strip()


# --------------------------------------------------------------------------
# Handwritten Text Extraction Section
# --------------------------------------------------------------------------
# EasyOCR is a deep learning based OCR engine that works offline.
# It can read cursive or messy handwriting better than Tesseract.
# We load the model once (globally) so that it doesn't reload on every Flask request.
# --------------------------------------------------------------------------


def extract_handwritten_text(image_path):
    """
    Extract handwritten (or mixed) text using EasyOCR.
    Works fully offline after first small model load.
    """
    try:
        # Step 1: Run EasyOCR reader on image
        results = reader.readtext(image_path, detail=0, paragraph=True)

        # Step 2: Combine detected lines into a single string
        text = " ".join(results)

        # Step 3: Return cleaned text
        return text.strip()

    except Exception as e:
        print(f"[Handwritten OCR Error]: {e}")
        return ""
