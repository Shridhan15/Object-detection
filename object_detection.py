from ultralytics import YOLO

# Load a pre-trained YOLOv8 model.
# Model knows how to recognize common everyday objects
# like people, cars, dogs, chairs, bottles, etc.
model = YOLO("yolov8n.pt")

def detect_objects(image_path):
    """
    Detect objects in an image using the YOLOv8 model.
    
    1. The model looks at the image and tries to spot known objects in it.
    2. For every object it finds, it assigns a "label" (like 'person', 'car', 'dog').
    3. We collect these label.
    
    Example:
        If the image has 3 people and 2 cars,
        the function will return:
        ["person", "car"]

    Returns a list of unique object labels detected in the image.
    """
    results = model(image_path)   # Run the YOLO model on the given image
    objects = []
    # Each detected box has a class ID; we convert that into a human-readable label
    for r in results[0].boxes.cls:
        label = results[0].names[int(r)]
        if label not in objects:
            objects.append(label)
    return objects
