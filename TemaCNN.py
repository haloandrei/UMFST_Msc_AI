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

import os
import shutil
import kagglehub

DEST_DIR = "filtered_apples/train"

SPECIES_MAP = {
    "Apple Braeburn": "Apple Braeburn",
    "Apple Crimson Snow": "Apple Crimson Snow",
    "Apple Golden 1": "Apple Golden",
    "Apple Golden 2": "Apple Golden",
    "Apple Golden 3": "Apple Golden",
    "Apple Granny Smith": "Apple Granny Smith",
    "Apple Pink Lady": "Apple Pink Lady",
    "Apple Red 1": "Apple Red",
    "Apple Red 2": "Apple Red",
    "Apple Red 3": "Apple Red",
    "Apple Red Delicious": "Apple Red Delicious",
    "Apple Red Yellow 1": "Apple Red Yellow",
    "Apple Red Yellow 2": "Apple Red Yellow"
}

def prepare_dataset():

    print("Fetching dataset via kagglehub...")
    dataset_path = kagglehub.dataset_download("moltean/fruits")
    print(f"Dataset located at: {dataset_path}")

    training_path = None
    for root, dirs, files in os.walk(dataset_path):
        if "Training" in dirs:
            training_path = os.path.join(root, "Training")
            break

    if not training_path:
        print("Error: Could not find a 'Training' folder inside the downloaded dataset.")
        return

    print(f"Found Training folder at: {training_path}")


    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
    os.makedirs(DEST_DIR)

    print("Copying and merging specific apple folders...")

    count = 0

    for folder_name in os.listdir(training_path):
        source_folder = os.path.join(training_path, folder_name)

        if os.path.isdir(source_folder) and folder_name in SPECIES_MAP:

            target_class = SPECIES_MAP[folder_name]
            target_folder = os.path.join(DEST_DIR, target_class)
            os.makedirs(target_folder, exist_ok=True)

            for image_file in os.listdir(source_folder):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_file = os.path.join(source_folder, image_file)

                    new_name = f"{folder_name}_{image_file}"
                    dst_file = os.path.join(target_folder, new_name)

                    shutil.copy2(src_file, dst_file)
                    count += 1

            print(f"Processed: {folder_name} -> {target_class}")

    print(f"Done! {count} images organized into '{DEST_DIR}'.")

if __name__ == "__main__":
    prepare_dataset()