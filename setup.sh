#make virtual environment
python -m venv env
#open virtual environment
source ./env/bin/activate
#find dependencies and install requirements 
pip install --upgrade pip
sudo apt-get update
#pip install pipreqs
#pipreqs src --savepath requirements.txt
pip install -r requirements.txt
#make env avaible in nb
python -m pip install ipykernel
python -m ipykernel install --user --name=env

#exit the virtual environment
deactivate