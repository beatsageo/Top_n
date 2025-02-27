source ./venv/bin/activate # activate the venv

mkdir ./temp
python -W ignore ./spectrogramGen.py -Wignore --inputPath $1 --outputPath ./temp # output all audio
python run_all_models.py --inputPath ./temp --outputPath ./tempOut.json --logPath ./predictLog.json
rm -r ./temp

cat ./tempOut.json

deactivate # turn off the venv, return the user to their normal shell environment
