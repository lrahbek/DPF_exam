import os 
import sys
import pathlib
import gdown
import pyarrow
import pandas as pd
import numpy as np
import pooch
import itertools
from pooch import DOIDownloader


def download_chorus(data_path = "../data/raw/chorus"):
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


def download_shormccarty(data_path = "../data/raw/shormccarty/state_ideology.tab"):
    """
    Download data: Shor, Boris, 2023, "Aggregate State Legislator Shor-McCarty Ideology Data, April 2023 update",
    https://doi.org/10.7910/DVN/WI8ERB, Harvard Dataverse.
    """
    downloader = pooch.DOIDownloader()
    #url = "doi:10.7910/DVN/AP54NE/shor mccarty 1993-2018 state aggregate data July 2020 release.tab"
    url = "doi:10.7910/DVN/WI8ERB/shor mccarty 1993-2020 state aggregate data April 2023 release.tab"

    downloader(url=url, output_file=data_path, pooch=None)
    return print(f"file saved correctly at {data_path}: {os.path.exists(data_path)}")


def clean_bills(bills, states_list, times_ranges_list):
    """ 
    Bills are subsetted according to the following: 
    - bills from states in 'states_list' are kept.
    - bills within time ranges from 'times_ranges_list' are kept.
    - bills where 'status' isn't avaible are removed 
    - bill duplicates are removed.

    'ncsl_metatopics' and 'ncsl_topics' are split into lists and a column 'pass' is defined based
    on the 'status' column. from the column 'bill_chamber', the chamber of the given bill is
    extracted (col: 'cha').

    The cleaned Bills dataframe is returned with the columns: "state_unified_bill_id", "pass",  
    "state", "ncsl_topics", "ncsl_metatopics", "year", "bill_chamber"
    """
    bills_sub = bills[bills["state"].isin(states_list)]       #subset states with lobby records
    bills_sub = bills_sub[bills_sub["status"].notna()]        #remove rows with no status metadata
    bills_sub.drop_duplicates(subset="state_unified_bill_id", #remove rows with duplicate bill ids
                              ignore_index=True, inplace=True)
    #define bill year: 
    bills_sub["year"] = bills_sub[                            
        "last_action_date"].str.split(r"-", expand=True)[0]   #get year from 'last_action_date'  
    bills_sub.loc[bills_sub[bills_sub["year"].isin(           #else:get year from '...bill_id'
        [None, "0000", "1969"])].index, "year"] = bills_sub[bills_sub["year"].isin(
            [None, "0000", "1969"])]["state_unified_bill_id"].str.split("_", expand=True)[3]
    bills_sub["year"] = bills_sub["year"].astype("int32")     #set as type int 

    #define indices of bills to keep (according to states_list and time_ranges)
    keep_ind = []
    for i, (state, span) in enumerate(zip(states_list, times_ranges_list)):
        i_ind = bills_sub[(bills_sub["state"] == state) & (bills_sub["year"].isin(span))].index.tolist()
        keep_ind = keep_ind + i_ind
    bills_sub = bills_sub.iloc[keep_ind]

    #define binary 'pass' col
    bills_sub["pass"] = bills_sub.index.map(bills_sub["status"].isin([4,5]).to_dict())
    #split ncsl_topics and metatopics into lists
    bills_sub = bills_sub.replace(to_replace={                                  #replace none with 'M' (so it can be made to list)
        "ncsl_metatopics":{None: list(["M"])}, "ncsl_topics":{None: list(["M"])}})
    bills_sub["ncsl_topics"] = bills_sub["ncsl_topics"].str.split("; ")         #split topics into lists 
    bills_sub["ncsl_metatopics"] = bills_sub["ncsl_metatopics"].str.split("; ") #split topics into lists 
    #extract bill chamber
    bills_sub["cha"] = bills_sub["bill_chamber"].apply(lambda x: "H" if x[0] == "H" or x[0] == "A" else "S")
    #keep relevant columns: 
    bills_sub = bills_sub[["state_unified_bill_id", "pass",  "state", "ncsl_topics", 
                           "ncsl_metatopics", "year", "bill_chamber", "cha"]]
    return bills_sub


def clean_positions(positions, bills_sub, states_list, times_ranges_list):
    """
    Positions are subsetted according to the following: 
    - positions from states in 'states_list' are kept.
    - positions within time ranges from 'times_ranges_list' are kept.
    - rows with bills that aren't in the bills_sub data are removed (these had status NA)

    numeric values in 'position_numeric' are replaces with str - to make it easier to pivot the df. 
    Returns relevant columns: "state_client_id", "state", "state_unified_bill_id", 
    "position_numeric", "year"
    """
    #keep only rows from states_list and time_ranges
    keep_ind = []
    for i, (state, span) in enumerate(zip(states_list, times_ranges_list)):
        i_ind = positions[(positions["state"] == state) & (positions["year"].isin(span))].index.tolist()
        keep_ind = keep_ind + i_ind
    positions_sub = positions.iloc[keep_ind]
    #remove rows with bills not in bills data: 
    positions_sub = positions_sub[positions_sub["state_unified_bill_id"].isin(bills_sub["state_unified_bill_id"])]
    #subset relevant columns 
    positions_sub = positions_sub[["state_client_id", "state", "state_unified_bill_id", "position_numeric", "year"]] 
    #replace numeric values with str in positions
    positions_sub = positions_sub.replace(to_replace={"position_numeric": {-1.0: "oppose", 0.0: "neutral", 1.0: "support"}})
    positions_sub.reset_index(drop=True, inplace=True)

    return positions_sub