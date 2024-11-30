"""


"""


import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# make sure to change the path to the correct path of mel spectrograms on your device
rock = r"C:\Users\drake\OneDrive\Desktop\Dataset_Spectrogram_v2\rock"

metal = r"C:\Users\drake\OneDrive\Desktop\Dataset_Spectrogram_v2\metal"

blues = r"C:\Users\drake\OneDrive\Desktop\Dataset_Spectrogram_v2\blues"

classical = r"C:\Users\drake\OneDrive\Desktop\Dataset_Spectrogram_v2\classical"

country = r"C:\Users\drake\OneDrive\Desktop\Dataset_Spectrogram_v2\country"

disco = r"C:\Users\drake\OneDrive\Desktop\Dataset_Spectrogram_v2\disco"

electronic = r"C:\Users\drake\OneDrive\Desktop\Dataset_Spectrogram_v2\electronic"

hiphop = r"C:\Users\drake\OneDrive\Desktop\Dataset_Spectrogram_v2\hiphop"

jazz = r"C:\Users\drake\OneDrive\Desktop\Dataset_Spectrogram_v2\jazz"

pop = r"C:\Users\drake\OneDrive\Desktop\Dataset_Spectrogram_v2\pop"

reggae = r"C:\Users\drake\OneDrive\Desktop\Dataset_Spectrogram_v2\reggae"

rnb = r"C:\Users\drake\OneDrive\Desktop\Dataset_Spectrogram_v2\rnb"




def load_spectrograms(rock, metal, blues, classical, country, disco, electronic, hiphop, jazz, pop, reggae, rnb):
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
    print("Loading the blues spectrograms...")
    for filename in os.listdir(blues):
        if filename.endswith(".jpg"):
            img_path = os.path.join(blues, filename)
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            x.append(img_array)
            y.append(2)
    print("Loading the classical spectrograms...")
    for filename in os.listdir(classical):
        if filename.endswith(".jpg"):
            img_path = os.path.join(classical, filename)
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            x.append(img_array)
            y.append(3)
    print("Loading the country spectrograms...")   
    for filename in os.listdir(country):
        if filename.endswith(".jpg"):
            img_path = os.path.join(country, filename)
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            x.append(img_array)
            y.append(4)
    print("Loading the disco spectrograms...")
    for filename in os.listdir(disco):
        if filename.endswith(".jpg"):
            img_path = os.path.join(disco, filename)
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            x.append(img_array)
            y.append(5)
    print("Loading the electronic spectrograms...")
    for filename in os.listdir(electronic):
        if filename.endswith(".jpg"):
            img_path = os.path.join(electronic, filename)
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            x.append(img_array)
            y.append(6)
    print("Loading the hiphop spectrograms...")
    for filename in os.listdir(hiphop):
        if filename.endswith(".jpg"):
            img_path = os.path.join(hiphop, filename)
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            x.append(img_array)
            y.append(7)
    print("Loading the jazz spectrograms...")
    for filename in os.listdir(jazz):
        if filename.endswith(".jpg"):
            img_path = os.path.join(jazz, filename)
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            x.append(img_array)
            y.append(8)
    print("Loading the pop spectrograms...")
    for filename in os.listdir(pop):
        if filename.endswith(".jpg"):
            img_path = os.path.join(pop, filename)
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            x.append(img_array)
            y.append(9)
    print("Loading the reggae spectrograms...")
    for filename in os.listdir(reggae):
        if filename.endswith(".jpg"):
            img_path = os.path.join(reggae, filename)
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            x.append(img_array)
            y.append(10)
    print("Loading the rnb spectrograms...")
    for filename in os.listdir(rnb):
        if filename.endswith(".jpg"):
            img_path = os.path.join(rnb, filename)
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            x.append(img_array)
            y.append(11)

    print(f"Loaded this many spectrograms: {len(x)}")
    return np.array(x), np.array(y)

if __name__ == "__main__":
    
    x, y = load_spectrograms(rock, metal, blues, classical, country, disco, electronic, hiphop, jazz, pop, reggae, rnb)
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


    



    