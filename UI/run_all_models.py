"""
This script is used to parse CLI arguments for --inputPath, --outputPath and --logPath with testIO.py logic
it loads multiple pre-trained models, loads spectrogram images from --inputPath and runs all models on each spectrogram
It also averages the models output for ensemble prediction as of now
It will then save the results to --outputPath and writes a log to --logPath

Usage:
python run_all_models.py  \
  --inputPath "C:/path/to/folder_of_spectrograms" \
  --outputPath "C:/path/to/output_predictions.json" \
  --logPath "C:/path/to/log_file.json"
"""

import sys
import os
import json
import numpy as np

from tensorflow.keras.models import load_model  # type:ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type:ignore


GENRES = [
    "blues", "classical", "country", "disco", "electronic",
    "hiphop", "jazz", "metal", "pop", "reggae",
    "rnb", "rock"
]


def parse_cli_args():
    argNum = len(sys.argv)
    currArg = 1

    input_path = "./"
    output_path = "./predictions.json"
    log_path = "./log.json"

    while currArg < (argNum - 1):
        match sys.argv[currArg]:
            case "--inputPath":
                input_path = sys.argv[currArg + 1]
                currArg += 1
            case "--outputPath":
                output_path = sys.argv[currArg + 1]
                currArg += 1
            case "--logPath":
                log_path = sys.argv[currArg + 1]
                currArg += 1
        currArg += 1

    return input_path, output_path, log_path


def load_spectrograms(input_path, target_size=(128, 128)):
    images = []
    filenames = []
    for fname in os.listdir(input_path):
        if fname.lower().endswith(".jpg"):
            full_path = os.path.join(input_path, fname)
            img = load_img(full_path, target_size=target_size)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            filenames.append(fname)
    return np.array(images), filenames


def main():
    input_path, output_path, log_path = parse_cli_args()

    model_paths = [
        r"C:\Users\drake\OneDrive\Desktop\Senior_software_II\saved models\improved_master_model_dataset_Spectrogram_Balanced_v3.keras",
        r"C:\Users\drake\OneDrive\Desktop\Senior_software_II\saved models\improved_master_model.keras",
        r"C:\Users\drake\OneDrive\Desktop\Senior_software_II\saved models\my_model.keras_v1_gtzan.keras",
        r"C:\Users\drake\OneDrive\Desktop\Senior_software_II\saved models\sec_to_last.keras",
        r"C:\Users\drake\OneDrive\Desktop\Senior_software_II\saved models\spectogram_v1_gtzan_model.h5"
    ]

    print("Loading models........")
    models = []
    for mp in model_paths:
        if not os.path.isfile(mp):
            print(f"Model not found: {mp}")
            continue
        print(f"Loading model: {mp}")
        models.append(load_model(mp))

    print(f"Loading spectrograms from: {input_path}")
    X, file_names = load_spectrograms(input_path)
    if len(X) == 0:
        print("No spectrograms found in input path")
        return

    print("Running models on spectrograms........")
    all_preds = []
    for model in models:
        preds = model.predict(X)
        all_preds.append(preds)

    if not all_preds:
        print("No predictions made")
        return

    avg_preds = np.mean(all_preds, axis=0)
    final_labels = np.argmax(avg_preds, axis=1)

    results = []
    for fname, label_idx, raw_conf in zip(file_names, final_labels, avg_preds):
        results.append({
            "filename": fname,
            "ensemble_label": int(label_idx),
            "predicted_genre": GENRES[label_idx],  # Added genre mapping
            "combined_probabilities": raw_conf.tolist()
        })

    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, indent=2)
    print(f"Predictions saved to {output_path}")

    log_data = {
        "inputPath": input_path,
        "outputPath": output_path,
        "loadedModels": model_paths,
        "numSpectrograms": len(X),
        "numPredictions": len(results)
    }

    with open(log_path, "w", encoding="utf-8") as log_f:
        json.dump(log_data, log_f, indent=2)
    print(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
