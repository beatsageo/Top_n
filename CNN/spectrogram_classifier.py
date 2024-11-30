"""
Code as of now // feel free to update and change the code as needed
only supporting the rock and metal directories in Alex v3 dataset

"""


import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# make sure to change the path to the correct path of mel spectrograms on your device
rock = r"C:\Users\drake\OneDrive\Desktop\Dataset_Spectrogram_v2\rock"

metal = r"C:\Users\drake\OneDrive\Desktop\Dataset_Spectrogram_v2\metal"

def load_spectrograms(rock, metal):
    x = []
    y = []
    #ignore the .mp3 in the name of the files in the dataset its still a .jpg ext...
    print("Loading the rock spectrograms...")
    for filename in os.listdir(rock):
        if filename.endswith(".jpg"):
            img_path = os.path.join(rock, filename)
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            x.append(img_array)
            y.append(0)
    print("Loading the metal spectrograms...")
    for filename in os.listdir(metal):
        if filename.endswith(".jpg"):
            img_path = os.path.join(metal, filename)
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            x.append(img_array)
            y.append(1)
    print(f"Loaded this many spectrograms: {len(x)}")
    return np.array(x), np.array(y)

if __name__ == "__main__":
    
    x, y = load_spectrograms(rock, metal)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


    #first convolution layer will take in spectrograms of size 128x128x3, with ReLU activation function
    #second layer uses 64 filters and a 3x3 filter size with ReLU activation function
    #converts 2d image data into 1D array
    #connected layer combines features that are detected by convolution layer

    model = keras.Sequential([
        keras.layers.Conv2D(32,3, activation='relu', input_shape=(128, 128, 3)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    #binary classification with sigmoid activation using binary_crossentropy
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=10)


    



    