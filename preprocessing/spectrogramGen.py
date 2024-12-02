import librosa
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt

print(os.getcwd())

# Get audio file directory
try:
    from_dir = input("Please input the name of the folder with your audio files.\nInsert '~' to automatially append the working directory.\n>")
    if from_dir == "":
        from_dir = "~/audio-files"
    if from_dir[0] == "~":
        from_dir = os.getcwd() + from_dir[1:]
    songs = os.listdir(from_dir)
    print(f"{from_dir} Directory Found")
except FileNotFoundError:
    print(f"Error 404: {from_dir} Directory Not Found")
    quit()

# Filter out txt files

# Get spectrogram file directory
try: 
    to_dir = input("Input a name for your spectrograms' destination directory.\nInsert '~' to automatically append the working directory.\n>")
    if to_dir == "":
        to_dir = "spec-files"
    if to_dir[0] == "~":
        to_dir = os.getcwd() + to_dir[1:]
    os.mkdir(to_dir)
    print(f"Made {to_dir} Directory")
except FileExistsError:
    print(f"{to_dir} Directory Found")

# Create and save spectrogram images
songs = list(song for song in songs if song.endswith((".au",".mp3")))
numSongs = len(songs)
print(f"Processing {numSongs} audio files...")
for i in range(0,numSongs):
    # Loading Bar
    percent = ("{0:." + str(1) + "f}").format(100 * ((i+1) / float(numSongs)))
    filled = int(25 * (i+1) // numSongs)
    bar = "█" * filled + '-' * (25 - filled)
    print(f'\r[{bar}]  {percent}%  {i+1}/{numSongs}', end = '\r')
    # Spectrogram Generation
    y, sr = librosa.load(os.path.join(from_dir,songs[i]))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.savefig(f"{os.path.join(to_dir,songs[i])}.jpg")
    plt.close()