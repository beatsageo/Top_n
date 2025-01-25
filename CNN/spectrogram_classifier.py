import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

GENRE_PATHS = {
    0: r"C:\Users\drake\OneDrive\Desktop\Senior Software II\Out_Spectrogram_Balanced\blues",
    1: r"C:\Users\drake\OneDrive\Desktop\Senior Software II\Out_Spectrogram_Balanced\classical",
    2: r"C:\Users\drake\OneDrive\Desktop\Senior Software II\Out_Spectrogram_Balanced\country",
    3: r"C:\Users\drake\OneDrive\Desktop\Senior Software II\Out_Spectrogram_Balanced\disco",
    4: r"C:\Users\drake\OneDrive\Desktop\Senior Software II\Out_Spectrogram_Balanced\electronic",
    5: r"C:\Users\drake\OneDrive\Desktop\Senior Software II\Out_Spectrogram_Balanced\hiphop",
    6: r"C:\Users\drake\OneDrive\Desktop\Senior Software II\Out_Spectrogram_Balanced\jazz",
    7: r"C:\Users\drake\OneDrive\Desktop\Senior Software II\Out_Spectrogram_Balanced\metal",
    8: r"C:\Users\drake\OneDrive\Desktop\Senior Software II\Out_Spectrogram_Balanced\pop",
    9: r"C:\Users\drake\OneDrive\Desktop\Senior Software II\Out_Spectrogram_Balanced\reggae",
    10: r"C:\Users\drake\OneDrive\Desktop\Senior Software II\Out_Spectrogram_Balanced\rnb",
    11: r"C:\Users\drake\OneDrive\Desktop\Senior Software II\Out_Spectrogram_Balanced\rock"
}


def load_spectrograms(genre_paths, target_size=(128,128)):
    x, y = [], []

    for label, folder_path in genre_paths.items():
        print(f"Loading the spectrograms for label {label} from: {folder_path}")
        for filename in os.listdir(folder_path):
            # if file ends with .jpg
            if filename.endswith(".jpg"):
                img_path = os.path.join(folder_path, filename)
                # load and convert the image to an array
                img = load_img(img_path, target_size=target_size)
                img_array = img_to_array(img) / 255
                x.append(img_array)
                y.append(label)
    x = np.array(x)
    y = np.array(y)
    print(f"The total spectrograms loaded: {len(x)}")
    return x, y

if __name__ == "__main__":
    
    #Loading Data
    x, y = load_spectrograms(GENRE_PATHS)
    #Training Data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


    model = keras.Sequential([
        keras.layers.Conv2D(32,3, activation='relu', input_shape=(128, 128, 3)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(12, activation='softmax')
    ])

    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=5)

    
    """

    Save model if necessary
    model.save('example.keras')
    
    """

