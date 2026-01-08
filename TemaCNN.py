%pip install ultralytics -q

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import os

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

model = YOLO('yolov8n.pt') 

def detect_and_show(image_path):
    results = model.predict(source=image_path, conf=0.5, save=False)
    
    result_img = results[0].plot()
    print (results[0].boxes) 
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(result_img_rgb)
    plt.axis('off')
    plt.show()
    
    return results

if IN_COLAB:
    print("Please upload an image...")
    uploaded = files.upload()

    filename = list(uploaded.keys())[0]

    detection_results = detect_and_show(filename)

else:
    image_path = 'test_image.jpg'

    if os.path.exists(image_path):
        detection_results = detect_and_show(image_path)
    else:
        print(f"Error: {image_path} doesn't exist.")
