#open the virtual environment 
source ./env/bin/activate

python src/data_preperation.py
python src/bill_classification.py

#exit environment 
deactivate