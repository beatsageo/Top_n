import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

DATA_PATH = "data_10.json"


def load_data(data_path):
    """Loads training dataset from json file."""
    with open(data_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y


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


def prepare_datasets(test_size, validation_size):
    """Loads data and splits into train, validation, test sets with normalization."""
    X, y = load_data(DATA_PATH)

    # Stratified splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=validation_size, stratify=y_train)

    # Normalize MFCC coefficients
    mean = np.mean(X_train, axis=(0, 1))
    std = np.std(X_train, axis=(0, 1)) + 1e-7  # Avoid division by zero
    X_train = (X_train - mean) / std
    X_validation = (X_validation - mean) / std
    X_test = (X_test - mean) / std

    # Add channel dimension
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


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

    # Output Layer
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model


def predict(model, X, y, z, top_n=5):
    """Predict a single sample using the trained model and show top N predicted classes.

    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target label index
    :param z: List of genre names corresponding to label indices
    :param top_n (int): Number of top predictions to show
    """
    # Add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...]  # Array shape (1, 130, 13, 1)

    # Perform prediction
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


def save_model(model, filename="music_genre_model.keras"):
    """Saves the trained model to a file."""
    model.save(filename)
    print(f"Model saved to {filename}")


def load_model(filename="music_genre_model.keras"):
    """Loads a saved model from a file."""
    model = keras.models.load_model(filename, custom_objects={"SpecAugment": SpecAugment})
    print(f"Model loaded from {filename}")
    return model


if __name__ == "__main__":
    # Prepare data
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: w for i, w in enumerate(class_weights)}

    # Build or load model
    model_filename = "music_genre_model.keras"
    try:
        model = load_model(model_filename)
        print("Loaded existing model for further training.")
    except:
        print("Building a new model.")
        model = build_model((X_train.shape[1], X_train.shape[2], 1))

    # Compile model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    # Train
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation),
                        epochs=100, batch_size=32, callbacks=callbacks, class_weight=class_weights)

    # Save model
    save_model(model, model_filename)

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nTest Accuracy: {test_acc:.4f}')
    plot_history(history)

    # Predict a sample
    genre_mappings = [
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock"
    ]
    X_to_predict = X_test[100]
    y_to_predict = y_test[100]
    predict(model, X_to_predict, y_to_predict, genre_mappings)
