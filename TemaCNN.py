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

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

IMG_HEIGHT = 100
IMG_WIDTH = 100
BATCH_SIZE = 32
EPOCHS = 3

def train_cnn():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DEST,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DEST,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names

    print(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

  
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    model.save("apple_species_model.keras")
    print("Done")

    # Visualize Results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.figure(figsize=(8, 8))
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training Results')
    plt.show()

    # We need to extract labels and predictions from the dataset
    y_true = []
    y_pred = []

    # Iterate over the validation dataset to get predictions
    for images, labels in val_ds:
        # Get true labels for this batch
        y_true.extend(labels.numpy())
        
        # Get predictions for this batch
        predictions = model.predict(images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        y_pred.extend(predicted_classes)

    # Compute the matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot the matrix using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix: Apple Species')
    plt.show()


train_cnn()



IMAGE_PATH = "apples.jpg"

SPECIES_MODEL_PATH = "apple_species_model.keras"

# apples are id 47 in yolo

SPECIES_CLASSES = [
    'Apple Braeburn',
    'Apple Crimson Snow',
    'Apple Golden',
    'Apple Granny Smith',
    'Apple Pink Lady',
    'Apple Red',
    'Apple Red Delicious',
    'Apple Red Yellow'
]

def analyze_apples(image_apple):
    yolo = YOLO('yolo12n.pt')
    results = yolo.predict(image_apple, classes=[47], verbose=False)

    cpu_boxes = results[0].boxes.data.cpu().numpy()
    del yolo

    print(f"apple boxes: {len(cpu_boxes)}")

    species_model = tf.keras.models.load_model(SPECIES_MODEL_PATH)

    original_img = cv2.imread(image_apple)

    output_img = original_img.copy()

    for box in cpu_boxes:
        x1, y1, x2, y2, conf, cls_id = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        apple_crop = original_img[y1:y2, x1:x2]

        input_img = cv2.resize(apple_crop, (100, 100))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        input_tensor = np.expand_dims(input_img, axis=0)


        predictions = species_model.predict(input_tensor, verbose=0)
        predicted_index = np.argmax(predictions)
        confidence_score = np.max(predictions)
        species_name = SPECIES_CLASSES[predicted_index]

        print(f"Apple at [{x1}, {y1}]: {species_name} (confidence: {confidence_score:.2f})")

        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output_img, species_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    cv2.imwrite("result_apples1.jpg", output_img)

if IN_COLAB
    uploaded = files.upload()

    filename = list(uploaded.keys())[0]

    analyze_apples(filename)

else:
    analyze_apples(IMAGE_PATH)