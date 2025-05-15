# install brew and, using that python 3.12, may wish to refine for unattended install later
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew update --force --quiet
brew install python@3.12

# get the project
git clone https://github.com/beatsageo/Top_n.git
cd ./UI/

# create a venv and install requirements into it for the project
python3.12 -m venv venv
source ./bin/activate
pip install -r ./requirements.txt
chmod +x ./findGenre
deactivate

# tell the user we're ready
echo "Download models and store them in /Top_n/UI/models, then run the findGenre at will"
