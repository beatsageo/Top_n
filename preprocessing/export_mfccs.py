import json
import os
import math
import librosa
import sys
import numpy as np

# python export_mfccs.py gtzan_dataset_split gtzan_dataset_split_mfccs test

# typical call python export_mfccs.py ../AudioTest ../OutTest/ test
DATASET_PATH = sys.argv[1]
OUTPUT_FOLDER = sys.argv[2]
JSON_PREFIX = "-" + sys.argv[3] + "-data"
LOG_PATH = "-" + sys.argv[3] + "-log.json"

GENRE_MAPPING_PATH = "genre_mapping.json"  # Path to global genre mapping
SAMPLE_RATE = 22050
VECTOR_DURATION = 30  # measured in seconds
SAMPLES_PER_VECTOR = SAMPLE_RATE * VECTOR_DURATION


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


def save_mfcc(dataset_path, OUTPUT_FOLDER, json_prefix, log_path, genre_mapping_path, num_mfcc=13, n_fft=2048, hop_length=512, SAMPLES_PER_VECTOR=66150, vector_limit=10000):
    """Extracts MFCCs from music dataset and saves them into segmented JSON files along with genre labels.
       Logs errors for corrupt files.

       :param dataset_path (str): Path to dataset
       :param output_folder (str): Path to store output JSON files
       :param json_prefix (str): Prefix for exported JSON files
       :param log_path (str): Path to log file for errors
       :param genre_mapping_path (str): Path to global genre mapping file
       :param num_mfcc (int): Number of MFCC coefficients to extract
       :param n_fft (int): FFT window size
       :param hop_length (int): Step size for FFT
       :param vectors_per_segment (int): Number of vectors per segment
       :param vector_limit (int): Number of segments per JSON file
    """
    # Ensure the output folder exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load or create genre mapping
    genre_mapping = load_or_create_genre_mapping(genre_mapping_path)

    # Dictionary to store extracted features
    data = {
        "mapping": genre_mapping,
        "labels": [],
        "mfcc": []
    }

    # Dictionary to log errors
    log = {
        "vectorCount": 0,
        "failCount": 0,
        "failed": []
    }

    numJSON = 1  # JSON file index

    # Calculate the number of MFCC vectors per segment
    num_mfcc_coefficient_per_vector = math.ceil(SAMPLES_PER_VECTOR / hop_length)  # 130 mfccs vectors per segment

    # Loop through all genre sub-folders
    for dirpath, _, filenames in os.walk(dataset_path):

        # Ensure we're processing a genre sub-folder level
        if dirpath != dataset_path:
            semantic_label = os.path.basename(dirpath)

            # Ensure genre is in the mapping
            if semantic_label not in genre_mapping:
                update_genre_mapping(genre_mapping_path, semantic_label)
                genre_mapping.append(semantic_label)  # Update in-memory mapping

            genre_index = genre_mapping.index(semantic_label)

            print(f"\nProcessing: {semantic_label} (Label: {genre_index})")

            # Process each audio file in genre sub-directory
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                try:
                    # Load audio file
                    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                    # Calculate the total number of vectors
                    track_duration = len(signal) / sample_rate  # Duration in seconds
                    num_vectors = int((track_duration // VECTOR_DURATION))  # Adjust dynamically
                    if track_duration % VECTOR_DURATION >= VECTOR_DURATION / 2:
                        num_vectors += 1

                    print(f"Processing {file_path}: {track_duration:.2f} into ({num_vectors} vectors)")

                    # Process all vectors of audio file
                    for d in range(num_vectors):

                        # Define vector boundaries
                        samples_per_vector = sample_rate * VECTOR_DURATION
                        start = samples_per_vector * d
                        finish = start + samples_per_vector

                        # Ensure we do not go beyond the actual track length
                        if finish > len(signal):
                            remaining = finish - len(signal)
                            padded_signal = np.pad(signal[start:finish], (0, remaining), mode='wrap')
                            mfcc = librosa.feature.mfcc(y=padded_signal, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                            print(f"    {d+1}: Padded Vector between {int(start/sample_rate)} sec. and {int(len(signal)/sample_rate)}")
                        else:
                            mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                            print(f"    {d+1}: Vector from {int(start/sample_rate)} sec. to {int(finish/sample_rate)} sec.")
                        
                        mfcc = mfcc.T

                        # Store only vectors with the expected number of MFCC coefficients
                        if len(mfcc) == num_mfcc_coefficient_per_vector:  # 13 MFCC coefficients per segment
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(genre_index)

                            # Keeping track of the number of vectors processed
                            log["vectorCount"] += 1

                            # Create a new JSON file when vector limit is reached
                            if log["vectorCount"] % vector_limit == 0 and log["vectorCount"] != 0:
                                json_filename = os.path.join(OUTPUT_FOLDER, f"{json_prefix}_{numJSON}.json")
                                print(f"Saved {json_filename}")
                                with open(json_filename, "w") as fp:
                                    json.dump(data, fp, indent=4)

                                    # Reset data and increment file counter
                                    data = {
                                        "mapping": genre_mapping,
                                        "labels": [],
                                        "mfcc": []
                                    }
                                    numJSON += 1

                except Exception as e:
                    log["failCount"] += 1
                    log["failed"].append({"file": file_path, "error": str(e)})
                    print(f"Error processing {file_path}: {e}")

    # Save final JSON file if it contains data
    if data["mfcc"]:
        json_filename = os.path.join(OUTPUT_FOLDER, f"{json_prefix}_{numJSON}.json")
        with open(json_filename, "w") as fp:
            json.dump(data, fp, indent=4)
        print(f"Saved final {json_filename}")

    # Save log file
    log_filename = os.path.join(OUTPUT_FOLDER, log_path)
    with open(log_filename, "w") as fp:
        json.dump(log, fp, indent=4)

    print(f"\nProcessing complete. {log['vectorCount']} vectors extracted.")
    print(f"Failed to process {log['failCount']} files. See {log_path} for details.")


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, OUTPUT_FOLDER, JSON_PREFIX, LOG_PATH, GENRE_MAPPING_PATH, num_mfcc=13, n_fft=1048, hop_length=512, SAMPLES_PER_VECTOR=SAMPLES_PER_VECTOR, vector_limit=1000)
