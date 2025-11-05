import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import warnings
import zipfile
import shutil

# --- Mount Google Drive ---
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Suppress warnings
warnings.filterwarnings('ignore')

# --- 1. Constants and Configuration ---

GDRIVE_ZIP_PATH = '/content/drive/MyDrive/AD_Dataset_compressed.zip'
LOCAL_ZIP_PATH = '/content/Data.zip'
LOCAL_UNZIP_DIR = '/content/Data'

print(f"Copying {GDRIVE_ZIP_PATH} to Colab's local storage...")
try:
    shutil.copyfile(GDRIVE_ZIP_PATH, LOCAL_ZIP_PATH)
    print("Copy complete.")
except FileNotFoundError:
    print(f"ERROR: File not found at {GDRIVE_ZIP_PATH}")
    print("Please update the 'GDRIVE_ZIP_PATH' variable to the correct path.")
    raise

print(f"Unzipping {LOCAL_ZIP_PATH} to {LOCAL_UNZIP_DIR}...")
if os.path.exists(LOCAL_UNZIP_DIR):
    shutil.rmtree(LOCAL_UNZIP_DIR)
os.makedirs(LOCAL_UNZIP_DIR)

with zipfile.ZipFile(LOCAL_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(LOCAL_UNZIP_DIR)
print("Unzipping complete.")

# --- Use the known correct data path ---
DATA_DIR = '/content/Data/AD Datasets/Data'
if not os.path.isdir(DATA_DIR):
    print(f"ERROR: The path {DATA_DIR} does not exist.")
    raise FileNotFoundError("Could not locate the data directory.")
    
print(f"Data directory successfully set to: {DATA_DIR}")
print(f"Contents of data directory: {os.listdir(DATA_DIR)}")
# ----------------------------------


# --- Model parameters ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 50
LEARNING_RATE = 0.0001
SEED = 42

# --- Model Save Path ---
BEST_MODEL_SAVE_PATH = 'resnet50_best_model.keras'

# --- 2. Load Data & Create 80/10/10 Splits ---
print("Loading and splitting data (80% train, 10% validation, 10% test)...")

# First, load the entire dataset, shuffled, in batches
full_dataset = image_dataset_from_directory(
    DATA_DIR,
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

# Get the total number of batches
total_batches_tensor = tf.data.experimental.cardinality(full_dataset)

# --- FIX: Convert TensorFlow Tensor to Python int ---
# Check if dataset size is known
if total_batches_tensor == tf.data.experimental.UNKNOWN_CARDINALITY:
    print("ERROR: Could not determine the total number of batches.")
    print("This can happen if the dataset is very large or being streamed.")
    raise ValueError("Dataset cardinality is unknown.")

# Convert the tensor to a numpy integer
total_batches = total_batches_tensor.numpy()
# -----------------------------------------------------

# Calculate the number of batches for each set
train_batches = int(total_batches * 0.8)
val_batches = int(total_batches * 0.1)
# The test set gets the remainder
test_batches = total_batches - train_batches - val_batches

# Handle small datasets: ensure val_batches and test_batches are at least 1
if val_batches == 0 and total_batches > train_batches:
    val_batches = 1
if test_batches == 0 and total_batches > train_batches + val_batches:
    test_batches = 1
# Recalculate train_batches if we adjusted
train_batches = total_batches - val_batches - test_batches


print(f"Total batches: {total_batches}")
print(f"Training batches: {train_batches}")
print(f"Validation batches: {val_batches}")
print(f"Test batches: {test_batches}")

# Split the dataset
# 80% for training
train_dataset = full_dataset.take(train_batches)
# 20% for val/test
val_and_test_dataset = full_dataset.skip(train_batches)
# 10% for validation
val_dataset = val_and_test_dataset.take(val_batches)
# 10% for test
test_dataset = val_and_test_dataset.skip(val_batches)

# Get class names from the full dataset (they are all the same)
class_names = full_dataset.class_names
print(f"Found classes: {class_names}")

# --- 3. Handle Severe Class Imbalance (CRITICAL STEP) ---
print("Calculating class weights from training set...")
train_labels = []
# Iterate over the new train_dataset to get its labels
for images, labels in train_dataset:
    train_labels.extend(labels.numpy())

train_labels = np.array(train_labels)

class_weights_array = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

# Keras needs the weights in a dictionary format
print(f"Mapping weights to class names: {class_names}")
class_weight_dict_mapped = {}
unique_labels_sorted = np.unique(train_labels)
for i, class_index in enumerate(unique_labels_sorted):
    class_weight_dict_mapped[class_index] = class_weights_array[i]
    
print(f"Calculated class weights: {class_weight_dict_mapped}")

# --- 4. Configure Dataset for Performance ---
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# --- 5. Optional: Add Data Augmentation ---
# --- FIX: Removed extra space in variable name ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# --- 6. Build the Transfer Learning Model (ResNet50) ---
print("Building model with pre-trained ResNet50 base...")

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)
x = tf.keras.applications.resnet50.preprocess_input(x)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)

# --- 7. Compile the Model ---
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- 8. Model Summary ---
print("Model Summary:")
model.summary()

# --- 9. NEW: Create ModelCheckpoint Callback ---
# This callback will save the model *only* if `val_accuracy` improves.
checkpoint_callback = ModelCheckpoint(
    filepath=BEST_MODEL_SAVE_PATH,  # File path to save the model
    monitor='val_accuracy',       # The metric to monitor
    mode='max',                   # We want to 'max'imize accuracy
    save_best_only=True,          # Only save the best model
    verbose=1                     # Print a message when saving
)

# --- 10. Train the Model ---
print("\n--- Starting Model Training (50 Epochs) ---")
print(f"Best model will be saved to {BEST_MODEL_SAVE_PATH}")

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    class_weight=class_weight_dict_mapped,
    callbacks=[checkpoint_callback] # <-- ADDED the callback
)

print("\n--- Training Complete ---")

# --- 11. NEW: Load and Evaluate the BEST Model on all 3 Datasets ---
print(f"Loading the best model saved at {BEST_MODEL_SAVE_PATH}...")
try:
    # Load the model that had the highest validation accuracy
    best_model = tf.keras.models.load_model(BEST_MODEL_SAVE_PATH)
    
    print("Evaluating the best model on the TRAINING set...")
    train_loss, train_accuracy = best_model.evaluate(train_dataset)
    
    print("Evaluating the best model on the VALIDATION set...")
    val_loss, val_accuracy = best_model.evaluate(val_dataset)

    print("Evaluating the best model on the (unseen) TEST set...")
    # Evaluate this best model on the test_dataset
    test_loss, test_accuracy = best_model.evaluate(test_dataset)

    print("\n--- Final Model Performance ---")
    print(f"Training Accuracy:   {train_accuracy:.4f} ({(train_accuracy * 100):.2f}%)")
    print(f"Validation Accuracy: {val_accuracy:.4f} ({(val_accuracy * 100):.2f}%)")
    print(f"Test Accuracy:       {test_accuracy:.4f} ({(test_accuracy * 100):.2f}%)")
    
    print("\n--- Final Model Loss ---")
    print(f"Training Loss:   {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Test Loss:       {test_loss:.4f}")

except Exception as e:
    print(f"\n--- Error during final evaluation ---")
    print(f"Could not load or evaluate the best model. Error: {e}")
    print("This can sometimes happen if training was stopped early")
    print("or if no model was saved (e.g., validation accuracy never improved).")
    print("Please check the training logs.")
