import cv2
import pytesseract
# Tesseract is an open-source OCR engine
# It can read text from images and return it.
# Tesseract must be  installed on your system for this to work.

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text(image_path):
    """
    Extract text from an image using pytesseract OCR.

    Steps in simple terms:
    1. Load the image from the given file path.
    2. Convert the image to grayscale (OCR works better on simpler images).
    3. Send the image to Tesseract OCR to "read" the text.
    4. Return the text as a Python string (empty string if nothing is found).

    Example:
        Image: [Picture with "Invoice #12345"]
        Output: "Invoice #12345"
    """
    # Step 1: Read the image
    img = cv2.imread(image_path)

    # Step 2:  Convert the image to grayscale (black and white shades only) 
    # so that Tesseract OCR can focus on the text shapes.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply Tesseract OCR to detect text
    text = pytesseract.image_to_string(gray)

    # Step 4: Return cleaned text
    return text.strip()
