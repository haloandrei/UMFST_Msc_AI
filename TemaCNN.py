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

# --- CONFIGURATION ---
BASE_DEST_DIR = "filtered_apples"
TRAIN_DEST = os.path.join(BASE_DEST_DIR, "train")
TEST_DEST = os.path.join(BASE_DEST_DIR, "test")

SPECIES_MAP = {
    "Apple Braeburn 1": "Apple Braeburn",
    "Apple Crimson Snow 1": "Apple Crimson Snow",
    "Apple Golden 1": "Apple Golden",
    "Apple Golden 2": "Apple Golden",
    "Apple Golden 3": "Apple Golden",
    "Apple Granny Smith 1": "Apple Granny Smith",
    "Apple Pink Lady 1": "Apple Pink Lady",
    "Apple Red 1": "Apple Red",
    "Apple Red 2": "Apple Red",
    "Apple Red 3": "Apple Red",
    "Apple Red Delicious 1": "Apple Red Delicious",
    "Apple Red Yellow 1": "Apple Red Yellow",
    "Apple Red Yellow 2": "Apple Red Yellow"
}

def process_subset(source_root, dest_root, subset_name):
    
    if not os.path.exists(dest_root):
        os.makedirs(dest_root)

    count = 0

    for folder_name in os.listdir(source_root):
        source_folder = os.path.join(source_root, folder_name)

        if os.path.isdir(source_folder) and folder_name in SPECIES_MAP:
            target_class = SPECIES_MAP[folder_name]
            target_folder = os.path.join(dest_root, target_class)
            if (folder_name == "Apple Braeburn"):
              print("Processing "+ folder_name + " into "+ SPECIES_MAP[folder_name])
            os.makedirs(target_folder, exist_ok=True)

            for image_file in os.listdir(source_folder):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_file = os.path.join(source_folder, image_file)

                    new_name = f"{folder_name}_{image_file}"
                    dst_file = os.path.join(target_folder, new_name)

                    shutil.copy2(src_file, dst_file)
                    count += 1

    print(f"Finished {subset_name}: {count} images organized into '{dest_root}'")

def prepare_dataset():
    # 1. Download
    print("Fetching dataset via kagglehub...")
    dataset_path = kagglehub.dataset_download("moltean/fruits")
    print(f"Dataset downloaded to: {dataset_path}")

    train_path = os.path.join(dataset_path, "fruits-360_100x100", "fruits-360", "Training")
    test_path = os.path.join(dataset_path, "fruits-360_100x100", "fruits-360", "Test")


    if os.path.exists(BASE_DEST_DIR):
        shutil.rmtree(BASE_DEST_DIR)

    process_subset(train_path, TRAIN_DEST, "TRAINING")
    process_subset(test_path, TEST_DEST, "TESTING")

prepare_dataset()
