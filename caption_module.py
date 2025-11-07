from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# BLIP = Bootstrapped Language Image Pretraining model
# It can describe images in simple natural language.

# Load model once globally to save time
print("🔹 Loading BLIP captioning model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("✅ BLIP model loaded successfully!")

def generate_caption(image_path):
    """
    Generate a simple caption for an image using the BLIP model.

    Example:
        Input: image of a man with a laptop
        Output: "a man sitting with a laptop"
    """
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")

        # Generate caption
        output = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(output[0], skip_special_tokens=True)

        # Clean and return
        return caption.strip().capitalize()
    except Exception as e:
        print(f"[Caption Generation Error]: {e}")
        return ""
