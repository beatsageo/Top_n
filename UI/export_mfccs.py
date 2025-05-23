"""
April 19, 2024
Joshua Kim New multiprocessing

Attempt to fix Alexander Walsh Reported Memory Leak

It should be writing the JSONs to whatever part of the disk the end user
directs it to write to - there's an --inputPath argument you feed it. Problem
was, it was mangling that (the /path/to/the/folder/of/JSONs was given twice -
/path/to/the/folder/of/JSONs//path/to/the/folder/of/JSONs/name.json instead of
only  /path/to/the/folder/of/JSONs/name.json)  and, so, was trying to write to
a location that did not exist which failed either silently, or put up too
small a warning for me to notice amidst all the other stuff.

Total time elapsed: 110.28 seconds
"""

import json
import gc
import os
import math
import librosa
import sys
import numpy as np
import psutil
from multiprocessing import Pool, cpu_count
import warnings
import time


warnings.filterwarnings("ignore")

# python export_mfccs.py gtzan_dataset_split gtzan_dataset_split_mfccs test electronic

# typical call python export_mfccs.py ../AudioTest ../OutTest/ test electronic


DATASET_PATH = sys.argv[1]
OUTPUT_FOLDER = sys.argv[2]
JSON_PREFIX = sys.argv[3] + "-data"
LOG_PATH = sys.argv[3] + "-log.json"
GENRE = sys.argv[4]
SAMPLE_RATE = 22050
VECTOR_DURATION = 30  # measured in seconds, in this final version, we don't do chunking
SAMPLES_PER_VECTOR = SAMPLE_RATE * VECTOR_DURATION


def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024  # in MB
    print(f"🔍 Memory Usage: {mem_mb:.2f} MB")


def load_or_create_genre_mapping(mapping_path):
    """Loads the global genre mapping file or creates it if it doesn't exist."""
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as fp:
            genre_data = json.load(fp)
            return genre_data["genres"]
    else:
        # If the file doesn't exist, create an empty mapping
        genre_list = []
        with open(mapping_path, "w") as fp:
            json.dump({"genres": genre_list}, fp, indent=4)
        return genre_list


def update_genre_mapping(mapping_path, new_genre):
    """Adds a new genre to the mapping if not already present and updates the file."""
    genres = load_or_create_genre_mapping(mapping_path)
    if new_genre not in genres:
        genres.append(new_genre)
        with open(mapping_path, "w") as fp:
            json.dump({"genres": genres}, fp, indent=4)
        print(f"Added new genre: {new_genre}")


def process_audio_file(args):
    """Processes a single audio file and returns its vectors or error info."""
    file_path, genre_index, num_mfcc, n_fft, hop_length, num_mfcc_coefficient_per_vector = args

    try:
        # Load audio file with efficient settings
        signal, sample_rate = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            res_type='kaiser_fast',
            mono=True
        )

        vectors = []
        track_duration = len(signal) / sample_rate
        num_vectors = int(track_duration // VECTOR_DURATION)
        if track_duration % VECTOR_DURATION >= VECTOR_DURATION / 2:
            num_vectors += 1

        for d in range(num_vectors):
            start = SAMPLES_PER_VECTOR * d
            finish = start + SAMPLES_PER_VECTOR

            if finish > len(signal):
                remaining = finish - len(signal)
                padded_signal = np.pad(signal[start:finish], (0, remaining), mode='wrap')
                mfcc = librosa.feature.mfcc(
                    y=padded_signal,
                    sr=sample_rate,
                    n_mfcc=num_mfcc,
                    n_fft=n_fft,
                    hop_length=hop_length
                ).T
            else:
                mfcc = librosa.feature.mfcc(
                    y=signal[start:finish],
                    sr=sample_rate,
                    n_mfcc=num_mfcc,
                    n_fft=n_fft,
                    hop_length=hop_length
                ).T

            if len(mfcc) == num_mfcc_coefficient_per_vector:
                vectors.append({
                    "file_name": file_path,
                    "genre": GENRE,
                    "mfcc": mfcc.tolist()
                })

        return {
            "file_path": file_path,
            "vectors": vectors,
            "error": None
        }
    except Exception as e:
        return {
            "file_path": file_path,
            "vectors": [],
            "error": str(e)
        }


def save_mfcc(dataset_path, output_folder, json_prefix, log_path,
              num_mfcc=13, n_fft=2048, hop_length=512,
              samples_per_vector=66150, vectors_per_file=1000):
    """Main function to process audio files using multiprocessing."""
    os.makedirs(output_folder, exist_ok=True)
    num_mfcc_coefficient_per_vector = math.ceil(samples_per_vector / hop_length)

    # Prepare all files to process with their arguments

    log = {
        "vectorCount": 0,
        "failCount": 0,
        "failed": []
    }

    for filenames in os.walk(dataset_path):
        for i, f in enumerate(filenames[2]):
            if f.endswith((".au",".mp3",".mp4","m4a",".webm")):
                file_path = os.path.join(dataset_path, f)

                result = process_audio_file((
                    file_path,
                    GENRE,
                    num_mfcc,
                    n_fft,
                    hop_length,
                    num_mfcc_coefficient_per_vector
                ))

                if result["error"]:
                    log["failCount"] += 1
                    log["failed"].append({
                        "file": result["file_path"],
                        "error": result["error"]
                    })
                    continue

                for j, vector in enumerate(result["vectors"]):
                    output_path = os.path.join(
                        output_folder,
                        f"{json_prefix}_{i}_{j}.json"
                    )
                    with open(output_path, "w") as fp:
                        json.dump(
                            vector, 
                            fp, 
                            indent=4
                        )
                    log["vectorCount"] += 1

    # Save log file
    with open(os.path.join(output_folder, log_path), "w") as fp:
        json.dump(log, fp, indent=4)

    print(f"\nProcessing complete. {log['vectorCount']} vectors extracted.")
    print(f"Failed to process {log['failCount']} files. See {log_path} for details.")


if __name__ == "__main__":

    start_time = time.time()

    save_mfcc(
        DATASET_PATH,
        OUTPUT_FOLDER,
        JSON_PREFIX,
        LOG_PATH,
        num_mfcc=13,
        n_fft=2048,
        hop_length=512,
        samples_per_vector=SAMPLES_PER_VECTOR,
        vectors_per_file=1000
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal time elapsed: {elapsed_time:.2f} seconds")
