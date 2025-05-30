#make virtual environment
python -m venv env
#open virtual environment
source ./env/bin/activate
#find dependencies and install requirements 
pip install --upgrade pip
sudo apt-get update
pip install -r requirements.txt
#make env available in nb
python -m pip install ipykernel
python -m pip install ipywidgets

python -m ipykernel install --user --name=env

#exit the virtual environment
deactivate