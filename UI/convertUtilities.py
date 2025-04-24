import json
import csv
import sys
import argparse
import os

GENRES = [
    "blues", "classical", "country", "disco", "electronic",
    "hiphop", "jazz", "metal", "pop", "reggae",
    "rnb", "rock"
]

#mapping each Genre to it's probabily, if N = -1 -> print all the probabilities
def mapping_prob(mapping = GENRES, pro_result = [], N = -1):
    gendict = {}
    #mapping genres and probabilities
    for index, gen in enumerate(mapping):
        gendict[gen] = pro_result[index]

    #sorting the dictionary DESC order based on the probability
    sorted_gendict = dict(sorted(gendict.items(), key=lambda x: x[1], reverse=True))

    str_format = ""
    if N > 0:
        sorted_gendict = dict(list(sorted_gendict.items())[:N])
        for key in sorted_gendict:
            str_format += key + ',' + str(sorted_gendict[key]) + ','

    return str_format[:(len(str_format)-2)]

#if N = -1 default, export all 12 genres. If write_to_file = false -> console, else -> write to a csv file with the same name
def convert_json_to_csv(filepath = "", N = -1, mapping = GENRES, write_to_file = False):
    #opening JSON file
    f = open(filepath)
    data = json.load(f)

    if not write_to_file:
        for record in data:
            print(record['filename'],',', mapping_prob(mapping, record['combined_probabilities'], N = N))
    else:
        csv_filename = filepath.replace(".json", ".csv")
        with open(csv_filename, 'w') as out_file:
            for record in data:
                filename = record['filename']
                probs = record['combined_probabilities']
                mapped = mapping_prob(mapping, probs, N=N)  # returns top N as a string
                line = f"{filename},{mapped}\n"
                out_file.write(line)

        print(f"CSV file saved to: {csv_filename}")

    #closing file
    f.close()

def read_files_in_directory(directory_path):
    """Reads and prints the number of correct and incorrect of a genre out.

    Args:
        directory_path: The path to the directory containing the files.
    """
    try:
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        if not files:
            print(f"No files found in directory: {directory_path}")
            return
        for file_name in files:
            file_path = os.path.join(directory_path, file_name)
            genre = file_name.split("-")[0] #get genre on file_name
            #open json file
            f = open(file_path)
            data = json.load(f)
            #correct and incorrect variable will set 0 at the begin of each genre
            correct = 0
            incorrect = 0
            for record in data:
                if genre == GENRES[record['ensemble_label']]:
                    correct += 1
                else: incorrect += 1
                
            print(f"{genre} \n correct: {correct} \n incorrect: {incorrect}")
            correct = 0
            incorrect = 0

    except FileNotFoundError:
         print(f"Error: Directory not found: {directory_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputFilePath", required=True, help="Path to the input JSON file")
    parser.add_argument("--topN", type=int, default=3, help="Number of top genres to extract")
    parser.add_argument("--w", type=bool, default=False, help="Write to CSV file or not")
    #remove the comment if you want a specific output file path
    #parser.add_argument("--outputFilePath", required=True, help="Path to save the output JSON")
    return  parser.parse_args()
    

if __name__ == "__main__":
    
    #Run or call the program like:
    #python convertUtilites.py --inputFilePath ./classical-results1.json --topN 4 --outputFilePath ./top4_output.json
    args = arg_parser()
    #print(args.inputFilePath, args.topN, args.outputFilePath)
    convert_json_to_csv(args.inputFilePath, N = args.topN, write_to_file=args.w)
    #print(args.inp, output_path, log_path)