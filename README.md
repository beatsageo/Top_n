
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

### Requirements
python 3.12

### Linux/MacOS
To install python 3.12, consider installing and using [Homebrew](https://brew.sh/). Once Homebrew is installed, run the following command to install python 3.12:
~~~
brew install python@3.12
~~~

Download the UI folder and run the following commands in that folder /path/to/the/folder/called/UI:
~~~
python3.12 -m venv ./venv
source ./venv/bin/activate
pip install -r ./requirements.txt
deactivate
~~~
The resultant venv folder will take up about 4.2 GiB of space. 
> Be aware that, if the path to the folder you downloaded the UI folder into has any :s in it, this will fail. To fix this error, relocate the UI folder

Download the other models - improved_master_model_dataset_Spectrogram_Balanced_v3.keras and sec_to_last.keras from TBD and place them in UI/models alongside final_model.keras

### Windows (Untested/Experimental)
If using a UNIX-like environment for Windows, like [Cygwin](https://cygwin.com/), then the Linux commands should be used instead, and findGenre should work, but one will have to install python 3.12 for one's environment. To install python 3.12 on Windows, visit [the Python website](https://www.python.org/downloads/windows/) to download and install the latest 3.12.x version of Python. This guide was written when that was 3.12.9, should extra errors crop up. Alternatively, install python 3.12 directly from the [Microsoft Store](https://apps.microsoft.com/detail/9ncvdn91xzqp?hl=en-us&gl=US).

Download the UI folder and run (using the Windows Terminal or cmd.exe, the commands are slightly different in PowerShell) the following commands in that folder, C:\path\to\the\folder\called\UI:
~~~
python3.12 -m venv .\venv
.\venv\Scripts\activate
pip install -r .\requirements.txt
deactivate
~~~
If PowerShell, use the following commands instead:
~~~
python3.12 -m venv .\venv
.\venv\Scripts\Activate.ps1
pip install -r .\requirements.txt
deactivate
~~~
The resultant venv folder will take up about 4.2 GiB? of space. 

Download the other models - improved_master_model_dataset_Spectrogram_Balanced_v3.keras and sec_to_last.keras from TBD and place them in UI\models alongside final_model.keras. Open run_all_models.py and change the strings of model_paths to be the paths to your models, it is currently set up for use on Linux/MacOS, not Windows.


Use findGenre_win, rather than findGenre on Windows. Alternatively, findGenre may work on Windows using [Git BASH](https://gitforwindows.org/)

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
