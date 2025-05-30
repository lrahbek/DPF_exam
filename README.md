### Interest Group Influence on Legislative Outcomes in the US: Machine Learning Analysis of the CHORUS Dataset

*31-05-2025*

Repository for the code made for the Data Science, Prediction, and Forecasting exam (2025) at the Master in Cognitive Science (Aarhus University)

**Repository Structure:**

```
├── data/
    ├── raw/
    └── preprocessed/
├── out/
    ├── MLP_eval/
    ├── figs/
    ├── models/
    └── ideology_summary.txt
├── src/
    ├── data_preperation.py
    ├── eda.ipynb
    ├── bill_classification.py
    ├── shap_values.py
    └── SHAP_plots.ipynb
├── .gitignore
├── LICENSE
├── README.md
├── activate.sh
├── requirements.txt
└── setup.sh
```
\
\
The data used is from the *CHORUS dataset*: 

@data{S3/RPU1QP_2024,  
author = {Hall, Galen and Basseches, Joshua and Bromley-Trujillo, Rebecca and Culhane, Trevor},  
publisher = {UNC Dataverse},  
title = {{Replication Data for: CHORUS: A New Dataset of State Interest Group Policy Positions in the United States}},  
UNF = {UNF:6:/8mtS0YGAlad3W5ah8gJ9g==},  
year = {2024},  
version = {V1},  
doi = {10.15139/S3/RPU1QP},  
url = {https://doi.org/10.15139/S3/RPU1QP}  
}

and the *Shor-McCarty dataset*: 

@data{DVN/WI8ERB_2023,  
author = {Shor, Boris},  
publisher = {Harvard Dataverse},  
title = {{Aggregate State Legislator Shor-McCarty Ideology Data, April 2023 update}},  
UNF = {UNF:6:iJ/Jyl7OB9WmSeq6mYH8Zw==},  
year = {2023},  
version = {V1},  
doi = {10.7910/DVN/WI8ERB},  
url = {https://doi.org/10.7910/DVN/WI8ERB}  
}
