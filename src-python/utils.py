import pandas as pd 
import numpy as np
import sys,os 
import getpass
import gc

USER = getpass.getuser()
DATASET_DIR = f"/home/{USER}/hdd/benchmarks"
ROOT_DIR = f"/home/{USER}/incit-spmv"
WORK_DIR = os.path.join(ROOT_DIR, "analysis-2")

def fullpath_to_key(path: str):
    s = path.replace(".mtx", "")
    s = s.split("/")
    group = s[-3]
    name = s[-2]
    return f"{group}--{name}"

def key_to_fullpath(key, mtx_dir):
    group, mtx = key.split("--")
    filename = os.path.join(mtx_dir, group, mtx, f"{mtx}.mtx")
    return filename


def read_and_clean_csv(raw_file):
    try:
        ret = pd.read_csv(raw_file)
        ret = ret.dropna().reset_index(drop=True)
        ret = ret.drop_duplicates().reset_index(drop=True)
    except FileNotFoundError:
        print(f"Not found : {raw_file}")
        ret = pd.DataFrame()
    return ret

def get_df_data(df, col: str, key: str):
    return pd.DataFrame.from_records(df[col].to_numpy()).set_index(df.index)[key]


