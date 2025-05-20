# install brew and, using that python 3.12, may wish to refine for unattended install later
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
../linuxbrew/.linuxbrew/bin/brew update --force --quiet
../linuxbrew/.linuxbrew/bin/brew install python@3.12

# add brew and whatnot to path, drawn from brew's own instructions, provided when you run the installer
echo >> ~/.bashrc
echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> ~/.bashrc
eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"

# get the project
git clone https://github.com/beatsageo/Top_n.git
cd ./Top_n/UI/

# create a venv and install requirements into it for the project
python3.12 -m venv venv
source ./venv/bin/activate
pip install -r ./requirements.txt
chmod +x ./findGenre
deactivate

# tell the user we're ready
echo "Download models and store them in /Top_n/UI/models, then run the findGenre at will"
