import os
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model as keras_load_model
from collections import Counter

import matplotlib.pyplot as plt

DATA_FOLDER_PATH = "gtzan_dataset_mfccs"  # Path to folder containing dataset JSON files
MODEL_PATH = "v0.7_model.keras"
NEW_MODEL_PATH = "v0.7_model.keras"  # Path to save the new model instead of replacing the existing one
GENRE_MAPPING_PATH = "genre_mapping.json"


class SpecAugment(keras.layers.Layer):
    """Applies Time and Frequency Masking to input spectrograms."""
    def __init__(self, time_mask=10, freq_mask=2, **kwargs):
        super().__init__(**kwargs)
        self.time_mask = time_mask
        self.freq_mask = freq_mask

    def call(self, inputs, training=None):
        if not training:
            return inputs
        # Time masking
        time_mask = tf.random.uniform([], 0, self.time_mask, dtype=tf.int32)
        time_start = tf.random.uniform([tf.shape(inputs)[0]], 0, tf.shape(inputs)[1] - time_mask, dtype=tf.int32)
        time_indices = tf.range(tf.shape(inputs)[1])[None, :, None, None]
        time_mask = (time_indices < time_start[:, None, None, None]) | (time_indices >= (time_start[:, None, None, None] + time_mask))
        inputs = tf.where(time_mask, inputs, 0.0)
        # Frequency masking
        freq_mask = tf.random.uniform([], 0, self.freq_mask, dtype=tf.int32)
        freq_start = tf.random.uniform([tf.shape(inputs)[0]], 0, tf.shape(inputs)[2] - freq_mask, dtype=tf.int32)
        freq_indices = tf.range(tf.shape(inputs)[2])[None, None, :, None]
        freq_mask = (freq_indices < freq_start[:, None, None, None]) | (freq_indices >= (freq_start[:, None, None, None] + freq_mask))
        inputs = tf.where(freq_mask, inputs, 0.0)
        return inputs


def build_model(input_shape):
    """Generates an enhanced CNN model with regularization and augmentation."""
    model = keras.Sequential()

    # Data Augmentation
    model.add(SpecAugment(time_mask=10, freq_mask=2, input_shape=input_shape))

    # Conv Block 1
    model.add(keras.layers.Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same'))
    model.add(keras.layers.SpatialDropout2D(0.1))

    # Conv Block 2
    model.add(keras.layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same'))
    model.add(keras.layers.SpatialDropout2D(0.1))

    # Conv Block 3
    model.add(keras.layers.Conv2D(128, (2,2), padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2), padding='same'))
    model.add(keras.layers.SpatialDropout2D(0.1))

    # Global Pooling and Dense Layers
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.3))

    # Load genre mapping
    with open(GENRE_MAPPING_PATH, "r") as fp:
        genre_data = json.load(fp)
        genres = genre_data["genres"]

    # Output Layer
    model.add(keras.layers.Dense(len(genres), activation='softmax'))
    return model


def train_model(model, data_folder, test_size, validation_size, batch_size=32, epochs=30, callbacks=None):
    """Trains model

    :param model: Model to train
    :param data_folder: Folder containing JSON files of MFCCs and labels   
    :param test_size: Percentage of data to use for testing
    :param validation_size: Percentage of data to use for validation
    :param batch_size: Batch size for training
    :param epochs: Number of epochs to train
    :param callbacks: List of callbacks for training
    """
    dataset_files = [f for f in os.listdir(data_folder) if f.endswith(".json")]

    for dataset_file in dataset_files:
        file_path = os.path.join(data_folder, dataset_file)
        print(f"\n🔹 Training on dataset: {dataset_file}")

        # Load dataset in chunks
        with open(file_path, "r") as fp:
            data = json.load(fp)

        X = np.array(data["mfcc"])  # MFCC features
        y = np.array(data["labels"])  # Labels

        print("Class distribution:", Counter(y))  # Check labels before splitting

        # Train-validation-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size, stratify=y_train)

        # Compute mean & std for normalization of MFCC coefficients
        mean = np.mean(X_train, axis=(0, 1))
        std = np.std(X_train, axis=(0, 1)) + 1e-7

        # Save normalization parameters for later use in inference
        np.save("mfcc_mean.npy", mean)
        np.save("mfcc_std.npy", std)
        print("Saved MFCC normalization parameters (mean & std)") 

        # Normalize the data
        X_train = (X_train - mean) / std
        X_validation = (X_validation - mean) / std
        X_test = (X_test - mean) / std

        # Train model on this batch
        history = model.fit(
            X_train, y_train,
            validation_data=(X_validation, y_validation),
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
        )

        # Evaluate on test set after each dataset
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
        print(f"\n✅ Finished training on {dataset_file} | Test accuracy: {test_acc:.4f}")

    print("\n🎉 Training complete for all datasets!")

    return history, X_test, y_test


def plot_history(history):
    """Plots accuracy/loss for training/validation set."""
    fig, axs = plt.subplots(2)
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    plt.show()


def save_model(model, filename=MODEL_PATH):
    """Saves the trained model to a file."""
    model.save(filename)
    print(f"Model saved to {filename}")


def load_trained_model(filename=MODEL_PATH):
    """Loads a saved model from a file."""
    model = keras.models.load_model(filename, custom_objects={"SpecAugment": SpecAugment})
    print(f"Model loaded from {filename}")
    return model


def predict(model, X, y, z, top_n=5):
    """Predict a single sample using the trained model and show top N predicted classes.
    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target label index
    :param z: List of genre names corresponding to label indices
    :param top_n (int): Number of top predictions to show
    """

    # Load saved mean & std for normalization
    mean = np.load("mfcc_mean.npy")
    std = np.load("mfcc_std.npy")

    # Apply normalization using saved parameters
    X = (X - mean) / std

    # Ensure X has correct shape: (130, 13)
    if len(X.shape) == 2:  
        X = np.expand_dims(X, axis=-1)  # Add channel dimension -> (130, 13, 1)

    X = np.expand_dims(X, axis=0)   # Ensure batch dimension is first

    print(f"Input shape before prediction: {X.shape}")

    prediction = model.predict(X)

    # Get predicted probabilities and the corresponding class indices
    predicted_probs = prediction[0]  # The first (and only) item, since batch size is 1
    predicted_indices = np.argsort(predicted_probs)[::-1]  # Sort indices in descending order of probabilities

    # Map the target label and predicted label to genres using the z
    target_genre = z[y]
    predicted_genre = z[np.argmax(predicted_probs)]  # The genre with highest probability

    # Print the target and predicted genres
    print(f"Target: {target_genre}, Predicted label: {predicted_genre}")

    # Print the top N predicted genres with their probabilities
    print(f"Top {top_n} predictions:")
    for i in range(min(top_n, len(predicted_probs))):  
        index = predicted_indices[i]
        print(f"{z[index]}: {predicted_probs[index]:.4f}")


def train_and_evaluate():
    """Train and evaluate the model"""

    if os.path.exists(MODEL_PATH):
        model = load_trained_model(MODEL_PATH)
        print("Loaded existing model.")

    else:
        print("No existing model found. Building a new model.")
        input_shape = (130, 13, 1)
        model = build_model(input_shape)

    # Compile model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(filepath="best_model.keras", monitor="val_loss", save_best_only=True, mode="min", verbose=1)
    ]

    # Train the model
    history, X_test, y_test = train_model(model, DATA_FOLDER_PATH, test_size=0.25, validation_size=0.1,
                                          batch_size=32, epochs=30, callbacks=callbacks)

    # Save the trained model
    save_model(model, NEW_MODEL_PATH)

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nTest Accuracy: {test_acc:.4f}')

    # Plot training history
    plot_history(history)


def run_prediction():
    """Run a single sample prediction by loading X, y from a dataset JSON file."""

    model = load_trained_model(MODEL_PATH)

    # Load genre mapping
    with open(GENRE_MAPPING_PATH, "r") as fp:
        genre_data = json.load(fp)
        genres = genre_data["genres"]

    # 🔹 Step 1: Load the dataset JSON file
    dataset_file = os.path.join(DATA_FOLDER_PATH, "gtzan_dataset_mfccs_1.json")  # Update with correct filename
    with open(dataset_file, "r") as fp:
        data = json.load(fp)

    # 🔹 Step 2: Pick a sample index
    sample_index = 100  # Adjust if needed

    # 🔹 Step 3: Extract X (MFCC features) and y (label)
    X_to_predict = np.array(data["mfcc"][sample_index])  # Convert MFCC list to NumPy array
    y_to_predict = data["labels"][sample_index]  # Get label

    print(f"Loaded sample {sample_index}: X shape {X_to_predict.shape}, y label {y_to_predict}")

    # 🔹 Step 4: Ensure X has the correct shape and predict
    predict(model, X_to_predict, y_to_predict, genres)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train/evaluate the model or run a prediction.")
    parser.add_argument("--mode", type=str, choices=["train", "predict"], required=True,
                        help="Choose to either 'train' the model or 'predict' a sample.")

    args = parser.parse_args()

    if args.mode == "train":
        train_and_evaluate()

    elif args.mode == "predict":
        run_prediction()
