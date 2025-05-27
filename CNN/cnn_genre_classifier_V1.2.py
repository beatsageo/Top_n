# cnn_genre_classifier_V1.2.py

import os
import json
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
JSON_FOLDER = "gtzan_dataset_mfccs_v1"  # Path where multiple JSON files are stored
BATCH_SIZE = 16
EPOCHS = 5  # Adjust as needed, higher epochs may yield better results
VALIDATION_SPLIT = 0.2
SEED = 42
INPUT_SHAPE = (130, 13, 1)  # (timesteps, mfcc coefficients, channels)


# ===================== DATA GENERATOR =====================
class MFCCDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_list, batch_size, input_shape, shuffle=True):
        self.file_list = file_list
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.data = self._load_all_metadata()
        self.on_epoch_end()

    def _load_all_metadata(self):
        metadata = []
        for file_path in self.file_list:
            with open(file_path, "r") as f:
                content = json.load(f)
                for i, label in enumerate(content["labels"]):
                    metadata.append((file_path, i, label))  # Each segment indexed
        return metadata

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, index):
        batch_meta = self.data[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = [], []

        for file_path, _, label in batch_meta:
            with open(file_path, "r") as f:
                content = json.load(f)
                i = random.randint(0, len(content["mfcc"]) - 1)
                mfcc = np.array(content["mfcc"][i])
                if mfcc.shape != self.input_shape[:2]:
                    continue  # skip invalid shapes
                X.append(mfcc)
                y.append(label)

        X = np.array(X)[..., np.newaxis]  # Add channel dimension
        y = tf.keras.utils.to_categorical(y, num_classes=self._get_num_classes())

        return X, y

    def _get_num_classes(self):
        with open(self.data[0][0], "r") as f:
            content = json.load(f)
            return len(content["mapping"])


# ===================== MODEL =====================
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),

        layers.Conv2D(128, (2, 2), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


# ===================== PLOTTING FUNCTION =====================
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()


# ===================== MAIN =====================
def main():
    # Collect all JSON file paths
    all_files = [os.path.join(JSON_FOLDER, f) for f in os.listdir(JSON_FOLDER) if f.endswith(".json")]
    if not all_files:
        raise Exception("No JSON files found in folder.")

    # Split file list for training and validation
    train_files, val_files = train_test_split(all_files, test_size=VALIDATION_SPLIT, random_state=SEED)

    # Load mapping to determine number of classes
    with open(train_files[0], "r") as f:
        num_classes = len(json.load(f)["mapping"])

    # Create generators
    train_gen = MFCCDataGenerator(train_files, BATCH_SIZE, INPUT_SHAPE)
    val_gen = MFCCDataGenerator(val_files, BATCH_SIZE, INPUT_SHAPE, shuffle=False)

    # Build and compile model
    model = build_model(INPUT_SHAPE, num_classes)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint("best_model.h5", save_best_only=True),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-6
        )
    ]

    # Train model
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=EPOCHS,
                        callbacks=callbacks)

    # Save model
    model.save("cnn_genre_classifier_V1.3.h5")
    print("✅ Model saved as cnn_genre_classifier_V1.3.h5")

    # Plot training history
    plot_training_history(history)
    print("📈 Training history saved as training_history.png")


if __name__ == "__main__":
    main()
