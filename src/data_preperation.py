import os 
import sys
import pathlib
import gdown
from tqdm import tqdm 
import pyarrow
import pandas as pd
import numpy as np
import pickle as pkl
import pooch
import itertools
from pooch import DOIDownloader
import re


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
    bills_sub["pass"] = np.where(bills_sub['pass'] == True, 'passed', 'failed')
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
    print("bills cleaned")
    return bills_sub

def clean_ideology(ideology, bills_sub, states_list):
    """
    Ideology data is subsetted according to the following: 
    - Data from states in 'states_list' are kept
    - Subset years (roughly)
    Column 'st' is renamed to 'state' to be able to merge with other data

    The data is split according to the chamber and the appended (on rows) - to create one row per
    combination of chamber, year and state. 

    The columns are kept: "state", "year", "*_chamber", "*_dem", "*_rep", "*_majority", "_*minority",
    "*_dem_mean", "*_rep_mean", "*_diffs", "*_distance"

    returns a df merges with bills_sub
    """
    ideology_sub = ideology[ideology["st"].isin(states_list)]                 #subset states  
    ideology_sub = ideology_sub[ideology_sub["year"].isin(range(2009, 2021))]#subset years
        
    ideology_sub = ideology_sub[['st', 'year', 'hou_chamber', 'sen_chamber', 'hou_dem', 'hou_rep',
    'hou_majority', 'hou_minority', 'hou_dem_mean', 'hou_rep_mean', 'sen_dem', 'sen_rep', 'sen_majority', 
    'sen_minority', 'sen_dem_mean', 'sen_rep_mean', 'h_diffs', 's_diffs', 'h_distance', 's_distance']]
    ideology_sub = ideology_sub.rename(columns={"st": "state"}) #rename state col
    #split df and rbind
    sta_sen = ideology_sub.filter(regex=r'sen|s_|year|state').reset_index(drop=True)
    sta_hou = ideology_sub.filter(regex=r'hou|h_|year|state').reset_index(drop=True)
    sta_sen.loc[0:72, "cha"] = ['S']*72  #add chamber col
    sta_hou.loc[0:72, "cha"] = ['H']*72  #add chamber col
    sta_sen.columns = sta_sen.columns.str.removeprefix("sen_").str.removeprefix("s_")
    sta_hou.columns = sta_hou.columns.str.removeprefix("hou_").str.removeprefix("h_")
    #concat!
    ideology_sub = pd.concat([sta_sen, sta_hou]) 
    #merge with bills sub
    bills_ide = bills_sub.merge(ideology_sub, on=["state", "year", "cha"])
    print("ideology cleaned")
    return bills_ide

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
    print("positions cleaned")
    return positions_sub

def clean_blocks(blocks, positions_sub, states_list):
    """ 
    Blocks are subsetted according to the following: 
    - Rows from states in 'states_list' are kept 
    - Only rows with 'client_id' are kept (not 'state_unified_bill_id')
    - 'block_1' is the only block assignment column used. 
    - When merged with positions_sub, the rows with 'state_client_ids' not represented in blocks 
    are removed, and vice versa (the rows with 'state_clients' not represented in)

    'entity_id' is renamed 'state_client_id' to be able to merge with other data. 
    The merged df position_blocks with the columns: "state_unified_bill_id", "state_client_id", 
    "position_numeric", "block_1" is returned 
    """
    blocks_sub = blocks.rename(columns={"entity_id": "state_client_id"})                #rename to state_client_id 
    blocks_sub = blocks_sub[blocks_sub["state"].isin(states_list)]                      #only states with lobby recs
    blocks_sub = blocks_sub[blocks_sub["state_client_id"].str.match(r"[A-Z][A-Z]_\d+")] #keep only rows with clients
    blocks_sub["block_1"] = blocks_sub["block_1"].astype(str)                           #change block_1 type to str
    #merge blocks_sub and positions_sub
    positions_blocks = positions_sub.merge(
        blocks_sub, how="left", on=["state_client_id", "state"], indicator=True)
    #remove rows where no blocks have been assigned 
    positions_blocks = positions_blocks[positions_blocks["_merge"] == "both"]
    positions_blocks = positions_blocks[["state_unified_bill_id", "state_client_id", 
        "position_numeric", "block_1"]].sort_values("state_unified_bill_id").reset_index(drop=True)
    print("blocks cleaned")
    return positions_blocks


def extract_block_counts(bills_ide, positions_blocks, 
                         out_path = "../data/preprocessed/block_array.pkl"):   
    #define output array and dimensions: 
    block1_ls = positions_blocks["block_1"].unique()      #unique block1 values 
    bill_ls = bills_ide["state_unified_bill_id"].unique() #unique bill ids 
    #emtpty array for block counts [support, neutral, oppose]
    block_array = np.zeros([len(bill_ls), len(block1_ls),3], dtype = np.int32) 
    print(f"number of unique bills: {len(bill_ls)}\nnumber of unique blocks: {len(block1_ls)}")
    print(f"shape of block array: {block_array.shape}")

    for i in tqdm(range(len(bill_ls))):
        bill = bill_ls[i]

        ind_bill = np.asarray(bill_ls == bill).nonzero()   # get index of bill in list
        #check if bill has been lobbied on: 
        block_counts = positions_blocks[positions_blocks[   # get block counts 
                "state_unified_bill_id"]==bill].value_counts(["position_numeric", "block_1"])
        if len(block_counts)>0:
            for j, (count, block) in enumerate(zip(block_counts.values, block_counts.index)):
                ind_block = np.asarray(block1_ls == block[1]).nonzero() #get index of the block
                #check which place to put the count: 
                if block[0] == 'support':
                    block_array[ind_bill, ind_block, 0] = count
                if block[0] == 'neutral':
                    block_array[ind_bill, ind_block, 1] = count
                if block[0] == 'oppose':
                    block_array[ind_bill, ind_block, 2] = count
        
        #if bill hasnt been lobbied on skip it
        else: pass

    with open(out_path, 'wb') as file:
        pkl.dump((block_array, block1_ls, bill_ls), file)
    
    return block_array, block1_ls, bill_ls


def main():
    #load data
    chorus_folder = "../data/raw/chorus"
    bills = pd.read_parquet(os.path.join(chorus_folder, "bills.parquet"))
    positions = pd.read_parquet(os.path.join(chorus_folder, "positions.parquet"))
    blocks = pd.read_parquet(os.path.join(chorus_folder, "block_assignments.parquet"))
    ideology = pd.read_csv("../data/raw/shormccarty/state_ideology.tab", sep='\t')
    #def subset rules
    states_lobby = ["IA", "MA", "NE", "NJ", "RI", "WI"] 
    time_ranges = [range(2009, 2021), range(2009, 2021), range(2010, 2021), 
               range(2014, 2021), range(2018, 2021), range(2009, 2021)]
    #clean bills
    bills_sub = clean_bills(bills, states_lobby, time_ranges)
    #clean ideology 
    bills_ide = clean_ideology(ideology, bills_sub, states_lobby)
    #clean positions
    positions_sub = clean_positions(positions, bills_sub, states_lobby, time_ranges)
    #clean blocks 
    positions_blocks = clean_blocks(blocks, positions_sub, states_lobby)

    block_array, block1_ls, bill_ls = extract_block_counts(bills_ide, positions_blocks)

    #split bills data and save 
    y = bills_ide["pass"]
    X = bills_ide[["state", "ncsl_metatopics", "cha", "chamber", "dem", "rep", "majority", 
                   "minority", "dem_mean", "rep_mean", "diffs", "distance"]]
    with open("../data/preprocessed/features.pkl", 'wb') as file:
        pkl.dump((X,y), file)


if __name__ == "__main__":
    main()