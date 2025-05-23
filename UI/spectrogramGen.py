import librosa
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt

import sys

argNum = len(sys.argv)
currArg = 1  # we skip the first argument, since it's just the command name
INPUT_PATH = "./"  # default values
OUTPUT_PATH = "./out.type"


while currArg < (argNum-1):
    match sys.argv[currArg]:
        case "--inputPath":
            from_dir = sys.argv[currArg+1]  # if we see --inputPath, then the next bit of data *is* the input path
            songs = os.listdir(from_dir)  # this is also our data
            currArg += 1  # and we move one more along, since we just processed two lines, --inputPath and whatever that is, not one
        case "--outputPath":
            to_dir = sys.argv[currArg+1]
            currArg += 1
    currArg += 1  # we've processed this argument, next!

# Create and save spectrogram images
songs = list(song for song in songs if song.endswith((".au",".mp3",".mp4","m4a",".webm")))
numSongs = len(songs)
print(f"Processing {numSongs} audio files...")
for i in range(0, numSongs):
    # Loading Bar
    percent = ("{0:." + str(1) + "f}").format(100 * ((i+1) / float(numSongs)))
    filled = int(25 * (i+1) // numSongs)
    bar = "█" * filled + '-' * (25 - filled)
    print(f'\r[{bar}]  {percent}%  {i+1}/{numSongs}', end='\r')
    # Spectrogram Generation
    y, sr = librosa.load(os.path.join(from_dir, songs[i]))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.savefig(f"{os.path.join(to_dir,songs[i])}.jpg")
    plt.close(fig)