
<p align="center">
<img src="/osulogo.jpg" hspace="100" alt="image" width="300" height="auto">
<img src="/osulogo.jpg" hspace="100" alt="image" width="300" height="auto">
</p>

# Top_n Genre Classification Neural Neural Network 

<p align="center">
<img src="/MIT-Neural-Networks-SL.gif" alt="image" width="300" height="auto">
</p>

## Contributors
- D. Chauhan
- N. Huffman
- J. Kim
- B. Nguyen
- V. To
- A. Walsh

## About
This project is the Senior Software Engineering Design Project for the team of contributors. The main function of this software is a genre prediction algorithim.  Users will submit audio files for genra classification and the program will return the most probable genra classification probability.

![UML](/UML.jpg)


## License

View the MIT [LICENSE](/LICENSE)

## Installation

  TBD probably an installer for WIN/Mac/Linux

## Usage

### CLI Man Page 

NAME 

findGenre – find the genre of a music audio clip 

SYNOPSIS 

findGenre [OPTION] ... [INPUT]... 

DESCRIPTION 

Finds the genre of a music audio clip or folder, INPUT, outputting the results of the analysis to standard output (unless the user pipes that into a file) in the form of comma-separated genre names and confidence values, like this: 

audioClipOneNoExt, classical, 0.8, jazz, 0.6, rock, 0.2, pony, 0.02 

audioClipTwoNoExt, rock 0.99, jazz, 0.12, pony. 0.1, classical, 0.007  

If no INPUT is provided, the command does nothing, unless another option overrides it. If non-music files are in INPUT, they are ignored. 

-b, --best-genres=GENRES 

Restricts the output to no more than GENRES possible genres. This is a maximum and is therefore meaningless if the limit is set sufficiently high. If not included, the model will provide confidence values for all genres. 

-v, --version 

Output version information, including an accuracy rating of the current model, and exits. This accuracy metric is the sum of the model’s confidence in its best pick for correctly identified songs, minus the sum of the model’s confidence in its best pick for incorrectly identified songs, all divided by the total number of songs tested. 

-h, --help 

Output this man page, and exit. 

The GENRES argument is an unsigned integer. 

FILE must be an audio file belonging to a list, to be specified later (and likely expanded over time as development resources are available). 

Exit Status 

0	if OK, 

1	if there are errors in the CNN, output may be unreliable, 

2	if there are errors in the CNN, no data is output, 

3	if some audio clip not found or read, results likely incomplete, 

4	if there are command errors (ie invalid command line arguments) 

### EXAMPLES 

findGenre –b=2 ./ > foo.txt 

Attempts to read each file as the audio file type indicated by its extension (.mp4, .mp3, .wav...). The files that can be read are processed by the model, and the output strings are dumped into foo.txt, in the specified format, but with no more than two genre confidence values per file. 

findGenre ./R\ U\ L\ E\ T\ H\ E\ W\ O\ R\ L\ D\ \(Gulf\ War\ edit\).mp4 

Reads the file R U L E T H E W O R L D (Gulf War edit).mp4 from the current directory and outputs to the terminal the model’s confidence values for all genres. 


## Links


## Contribute!
  see [Contributing](/Contributing.md)
