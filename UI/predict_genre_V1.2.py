import os
import json
import numpy as np
import tensorflow as tf
import sys

# Constants
GENRE_LABELS = [
    "blues", "classical", "country", "disco", "electronic", "hiphop",
    "jazz", "metal", "pop", "reggae", "rnb", "rock"
]


def load_mfcc(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return np.array(data["mfcc"])


EXPECTED_MFCC_SHAPE = (336, 13)  # adjust based on training

EXPECTED_NUM_FRAMES = 130
NUM_MFCC = 13


def pad_or_trim_mfcc(mfcc):
    mfcc = np.array(mfcc)
    if mfcc.shape[0] > EXPECTED_NUM_FRAMES:
        mfcc = mfcc[:EXPECTED_NUM_FRAMES]
    elif mfcc.shape[0] < EXPECTED_NUM_FRAMES:
        pad_width = EXPECTED_NUM_FRAMES - mfcc.shape[0]
        pad = np.zeros((pad_width, NUM_MFCC))
        mfcc = np.vstack((mfcc, pad))
    return mfcc


def predict_genre(model, mfcc):
    mfcc = pad_or_trim_mfcc(mfcc)             # shape: (130, 13)
    mfcc = np.reshape(mfcc, (1, 130, 13, 1))   # shape: (1, 130, 13, 1)
    prediction = model.predict(mfcc, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    predicted_genre = GENRE_LABELS[predicted_index]
    return predicted_index, predicted_genre, prediction.tolist()


def main(model_path, json_dir):
    model = tf.keras.models.load_model(model_path)
    output_results = []

    for filename in sorted(os.listdir(json_dir)):
        if not filename.endswith(".json"):
            continue
        file_path = os.path.join(json_dir, filename)
        try:
            mfcc = load_mfcc(file_path)
            index, genre, probs = predict_genre(model, mfcc)
            output_results.append({
                "filename": filename,
                "ensemble_label": index,
                "predicted_genre": genre,
                "combined_probabilities": probs
            })
            print(f"Processed {filename}", file=sys.stderr)
        except Exception as e:
            print(f"Error processing {filename}: {e}", file=sys.stderr)

    # Output JSON array to stdout
    json.dump(output_results, sys.stdout, indent=4)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict_genre_V1.2.py model.h5 path_to_json_dir", file=sys.stderr)
        sys.exit(1)

    model_path = sys.argv[1]
    json_dir = sys.argv[2]
    main(model_path, json_dir)
