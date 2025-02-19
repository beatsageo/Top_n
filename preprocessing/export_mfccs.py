import json
import os
import math
import librosa

DATASET_PATH = "gtzan_dataset"
JSON_PREFIX = os.path.basename(DATASET_PATH) + "_mfccs"
LOG_PATH = JSON_PREFIX + "_log.json"
GENRE_MAPPING_PATH = "genre_mapping.json"  # Path to global genre mapping
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


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


def save_mfcc(dataset_path, json_prefix, log_path, genre_mapping_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5, segment_limit=10000):
    """Extracts MFCCs from music dataset and saves them into segmented JSON files along with genre labels.
       Logs errors for corrupt files.

       :param dataset_path (str): Path to dataset
       :param json_prefix (str): Prefix for exported JSON files
       :param log_path (str): Path to log file for errors
       :param genre_mapping_path (str): Path to global genre mapping file
       :param num_mfcc (int): Number of MFCC coefficients to extract
       :param n_fft (int): FFT window size
       :param hop_length (int): Step size for FFT
       :param num_segments (int): Number of segments per track
       :param segment_limit (int): Number of segments per JSON file
    """

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
        "segmentCount": 0,
        "failCount": 0,
        "failed": []
    }

    numJSON = 1  # JSON file index

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

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

                    # Process all segments of audio file
                    for d in range(num_segments):
                        log["segmentCount"] += 1

                        # Create a new JSON file every `segment_limit` segments
                        if log["segmentCount"] % segment_limit == 0:
                            json_filename = f"{json_prefix}_{numJSON}.json"
                            with open(json_filename, "w") as fp:
                                json.dump(data, fp, indent=4)
                            print(f"Saved {json_filename}")

                            # Reset data and increment file counter
                            data = {
                                "mapping": genre_mapping,
                                "labels": [],
                                "mfcc": []
                            }
                            numJSON += 1

                        # Define segment boundaries
                        start = samples_per_segment * d
                        finish = start + samples_per_segment

                        # Extract MFCCs
                        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                        mfcc = mfcc.T

                        # Store only segments with the expected number of MFCC vectors
                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(genre_index)
                            print("{}, segment: {}".format(file_path, d + 1))

                except Exception as e:
                    log["failCount"] += 1
                    log["failed"].append({"file": file_path, "error": str(e)})
                    print(f"Error processing {file_path}: {e}")

    # Save final JSON file if it contains data
    if data["mfcc"]:
        json_filename = f"{json_prefix}_{numJSON}.json"
        with open(json_filename, "w") as fp:
            json.dump(data, fp, indent=4)
        print(f"Saved final {json_filename}")

    # Save log file
    with open(log_path, "w") as fp:
        json.dump(log, fp, indent=4)

    print(f"\nProcessing complete. {log['segmentCount']} segments extracted.")
    print(f"Failed to process {log['failCount']} files. See {log_path} for details.")


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PREFIX, LOG_PATH, GENRE_MAPPING_PATH, num_segments=10, segment_limit=10000)
