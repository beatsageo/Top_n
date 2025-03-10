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
MODEL_PATH = "gtzan_model.keras"
NEW_MODEL_PATH = "gtzan_model.keras"  # Path to save the new model instead of replacing the existing one
GENRE_MAPPING_PATH = "genre_mapping.json"
BEST_MODEL_PATH = "best_model.keras"


def load_or_create_genre_mapping():
    """Loads the global genre mapping file or creates it if it doesn't exist."""
    if os.path.exists(GENRE_MAPPING_PATH):
        with open(GENRE_MAPPING_PATH, "r") as fp:
            return json.load(fp)["genres"]
    else:
        with open(GENRE_MAPPING_PATH, "w") as fp:
            json.dump({"genres": []}, fp, indent=4)
        return []


def update_genre_mapping(new_genres):
    """Ensures new genres are added to the global mapping."""
    genres = load_or_create_genre_mapping()
    updated = False
    for genre in new_genres:
        if genre not in genres:
            genres.append(genre)
            updated = True
    if updated:
        with open(GENRE_MAPPING_PATH, "w") as fp:
            json.dump({"genres": genres}, fp, indent=4)
    return genres


# Build Improved CNN Model
def build_model(input_shape, num_classes):
    model = keras.Sequential()

    # Conv Block 1 + SE
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.002)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D((3,3), strides=(2, 2), padding='same'))
    model.add(keras.layers.SpatialDropout2D(0.1))

    # Conv Block 2 + SE
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.002)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.SpatialDropout2D(0.1))

    # Conv Block 3 + SE
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.002)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.SpatialDropout2D(0.2))

    # Fully Connected Layers
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.002)))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.002)))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    return model


# Load and Combine Data from All Genres
def load_combined_data(DATA_FOLDER_PATH):
    X, y = [], []
    dataset_files = [f for f in os.listdir(DATA_FOLDER_PATH) if f.endswith(".json")]

    for dataset_file in dataset_files:
        file_path = os.path.join(DATA_FOLDER_PATH, dataset_file)
        with open(file_path, "r") as fp:
            data = json.load(fp)

        X.extend(data["mfcc"])
        y.extend(data["labels"])

    X = np.array(X)[..., np.newaxis]
    y = np.array(y)

    return X, y


def compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )


def train_model(model, X_train, y_train, validation_split, batch_size, epochs, class_weight_dict):
    """Trains model

    :param model: Model to train
    :param test_size: Percentage of data to use for testing
    :param validation_size: Percentage of data to use for validation
    :param batch_size: Batch size for training
    :param epochs: Number of epochs to train
    :param callbacks: List of callbacks for training
    """

    callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(filepath=BEST_MODEL_PATH, monitor="val_loss", save_best_only=True, mode="min", verbose=1)
        ]

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    return history


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
    model = keras.models.load_model(filename)
    print(f"Model loaded from {filename}")

    return model


def predict(top_n=5):
    """Predict a single sample using the trained model and show top N predicted classes.
    :param top_n (int): Number of top predictions to show
    """

    # Load the trained model
    model = load_trained_model(MODEL_PATH)

    # Load the combined data
    X, y = load_combined_data(DATA_FOLDER_PATH)

    # Load the genre mapping
    with open("genre_mapping.json", "r") as fp:
        genre_mapping = json.load(fp)["genres"]

    # Predict on a single sample
    sample_index = 5000  # Replace with the index of the sample you want to predict
    sample = X[sample_index]
    true_label = y[sample_index]

    # Expand dimensions to match the model's input shape
    sample = np.expand_dims(sample, axis=0)

    # Make a prediction
    predictions = model.predict(sample)
    predicted_probs = predictions[0]  # Get the probabilities for the single sample

    # Get the top N predicted genres and their probabilities
    top_n_indices = np.argsort(predicted_probs)[::-1][:top_n]  # Indices of top N predictions
    top_n_genres = [genre_mapping[i] for i in top_n_indices]  # Map indices to genre names
    top_n_probabilities = predicted_probs[top_n_indices]  # Get the corresponding probabilities

    # Map the true label to a genre
    true_genre = genre_mapping[true_label]

    # Print the true genre and top N predicted genres with their probabilities
    print(f"True Genre: {true_genre}")
    print(f"Top {top_n} predicted genres:")
    for genre, prob in zip(top_n_genres, top_n_probabilities):
        print(f"{genre}: {prob:.4f}")


def train_and_evaluate():
    """Train and evaluate the model"""

    X, y = load_combined_data(DATA_FOLDER_PATH)

    # Stratified splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    # Compute class weights to handle imbalance
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    num_classes = len(set(y_train))

    if os.path.exists(MODEL_PATH):
        model = load_trained_model(MODEL_PATH)
        print("Loaded existing model.")

    else:
        print("No existing model found. Building a new model.")
        model = build_model(X_train.shape[1:], num_classes)
        compile_model(model)

    # Train the model
    history = train_model(model, X_train, y_train, validation_split=0.2,
                                        batch_size=16, epochs=50, class_weight_dict=class_weight_dict)

    # Evaluate on test set after each dataset
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\n✅ Finished training | Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    # Save the trained model after each dataset
    save_model(model, NEW_MODEL_PATH)

    print("\n🎉 Training complete for all datasets!")

    # Plot training history
    plot_history(history)


if __name__ == "__main__":

    """Using CLI to choose between training and prediction.
    In cli, run the following commands:
    To train model: 'python cnn_genre_classifier_V0.9.py --mode train'
    To predict model: 'python cnn_genre_classifier_V0.9.py --mode predict'
    """

    # parser = argparse.ArgumentParser(description="Train/evaluate the model or run a prediction.")
    # parser.add_argument("--mode", type=str, choices=["train", "predict"], required=True,
    #                     help="Choose to either 'train' the model or 'predict' a sample.")

    # args = parser.parse_args()

    # if args.mode == "train":
    #     train_and_evaluate()

    # elif args.mode == "predict":
    #     run_prediction()

    # train_and_evaluate()
    predict()
