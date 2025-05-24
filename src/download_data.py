import os 
import pathlib
import gdown
import pyarrow
import pandas as pd
import numpy as np
import itertools
from pooch import DOIDownloader


def download_chorus(data_path = "data/chorus"):
    """ 
    Download Chorus data: Hall, Galen, Joshua Basseches, Rebecca Bromley-Trujillo, and Trevor Culhane. 2023. 
    "CHORUS: A new dataset of state interest group policy positions in the United States." State Politics & 
    Policy Quarterly. Forthcoming 2023.
    """
    path = pathlib.Path(data_path)
    path.mkdir(exist_ok=True)
    data_url = "https://drive.google.com/drive/folders/1JLxwurbx0ys4DUDB2o-WCtWWsjisVi8L?usp=sharing"
    gdown.download_folder(data_url, output = data_path)
    return print(f"Data downloaded from google drive to {data_path}")


def download_shormccarty(data_path = "data/shormccarty/state_ideology.tab"):
    """
    Download data: Shor, Boris, 2020, "Aggregate State Legislator Shor-McCarty Ideology Data, 
    July 2020 update", https://doi.org/10.7910/DVN/AP54NE, Harvard Dataverse. 
    """
    downloader = pooch.DOIDownloader()
    url = "doi:10.7910/DVN/AP54NE/shor mccarty 1993-2018 state aggregate data July 2020 release.tab"
    downloader(url=url, output_file=data_path, pooch=None)
    return print(f"file saved correctly at {data_path}: {os.path.exists(data_path)}")