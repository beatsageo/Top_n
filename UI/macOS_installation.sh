#!/bin/bash

# Install Homebrew if not already installed
if ! command -v brew &> /dev/null; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Update brew and install python 3.12
brew update
brew install python@3.12

# Add Python 3.12 to your PATH (only needed for current shell; to persist, add to ~/.zshrc)
export PATH="$(brew --prefix python@3.12)/bin:$PATH"

# Clone the project
git clone https://github.com/beatsageo/Top_n.git
cd Top_n/UI/

# Set up virtual environment
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Make the findGenre script executable
chmod +x findGenre

# Deactivate venv
deactivate
