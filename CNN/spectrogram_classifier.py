import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def auto_genre_paths(root_dir):
    """
    auto detects all subdirectories inside root_dir
    and map each to a numeric label, based on sorted folder names.
    """
    subdirs = [
        name for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    ]
    subdirs.sort()  

  
    return {
        i: os.path.join(root_dir, folder_name)
        for i, folder_name in enumerate(subdirs)
    }

def load_spectrograms(genre_paths, target_size=(128,128)):
    x, y = [], []
    for label, folder_path in genre_paths.items():
        print(f"Loading spectrograms for label {label} from: {folder_path}")
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                img_path = os.path.join(folder_path, filename)
                img = load_img(img_path, target_size=target_size)
                img_array = img_to_array(img) / 255.0  
                x.append(img_array)
                y.append(label)
    x = np.array(x)
    y = np.array(y)
    print(f"Total spectrograms loaded: {len(x)}")
    return x, y

if __name__ == "__main__":
    root_dir = r"C:\Users\drake\OneDrive\Desktop\Senior_software_II\Dataset_Spectrogram_Balanced_v3.0c"

  
    GENRE_PATHS = auto_genre_paths(root_dir)


    x, y = load_spectrograms(GENRE_PATHS)

   
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(len(GENRE_PATHS), activation='softmax')
    ])


    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        batch_size=32,
        epochs=5
    )


    """
    model.save("master_model.h5")
    
    """
    
