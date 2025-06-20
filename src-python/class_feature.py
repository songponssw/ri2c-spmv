# import pandas as pd
# import numpy as np
# import os, sys
#
# ROOT_DIR = "/home/thor/incit-spmv"
# WORK_DIR = os.path.join(ROOT_DIR, "analysis-2")

from utils import *


# def get_df_data(df, col: str, key: str):
#     return pd.DataFrame.from_records(df[col].to_numpy()).set_index(df.index)[key]
#

def calculate_offset_density(k_dict, N):
    l = []
    for k, nnz in k_dict.items():
        total = N - abs(k)
        p = round((nnz / total), 2)
        v = (k, p)
        l.append(v)
    return l


# This is criteria
def is_k_density_above_threshold(l, threshold):
    arr = np.array([v for _, v in l])
    return np.all(arr > threshold)


def filter_blocks(row):
    threshold = 0.0001 * (row["N"] // 10) * (row["N"] // 10)
    return [1 if val >= threshold else 0 for val in row["blocks_10"]]

def get_row_spread(blocks,threshold=1, N=1):
    M = 10
    # treshold value must deliveate by N
    width_per_block = N // M
    threshold = (width_per_block * width_per_block )* 0.001  # Equal 0.1%. The mean density is 0.5%
    
    
    ret = [0]*M
    for i in range(M):
        start, end = i*M, i*M+M
        l = blocks[start:end]
        count = sum(1 for x in l if x > threshold)
        ret[i] = count
    return ret 


def get_col_spread(blocks,threshold=1, N=1):
    M = 10
    width_per_block = N // M
    threshold = (width_per_block * width_per_block )* 0.001
    ret = [0]*M
    for j in range(M):
        i_list = [i*M+j for i in range(M)]
        # print (i_list)
        # start, end = i*M, i*M+M
        l = [blocks[idx] for idx in i_list]
        # print (l)
        count = sum(1 for x in l if x > threshold)
        ret[j] = count
    return ret 

def has_big_shift(arr, threshold=8):
    arr = np.array(arr)
    diffs = np.abs(np.diff(arr))  # Compute absolute differences between consecutive values
    return np.any(diffs >= threshold)  #

def get_long_spread(lst):
    arr = np.array(lst)
    if int(np.sum(arr == 10)) == 1 :
        return True 
    return False

if __name__ == "__main__":
    print("Calculate class features")
    dataset = sys.argv[1]
    output_file = os.path.join(WORK_DIR, "pkl", f"{dataset}-class_features.pkl")

    # kdensity
    # require k_dist
    kdist_pkl = os.path.join(WORK_DIR, "pkl", f"{dataset}-k_dist.pkl")
    submtx_pkl = os.path.join(WORK_DIR, "pkl", f"{dataset}-submatrices.pkl")

    print("Join everything in to data")
    data = pd.DataFrame()
    for f in [kdist_pkl, submtx_pkl]:
        tmp_df = pd.read_pickle(f)
        if data.empty:
            data = tmp_df
            continue
        data = data.join(tmp_df)

    features_pkl = os.path.join(WORK_DIR, "pkl", f"{dataset}-features.pkl")
    features_df = pd.read_pickle(features_pkl)

    feature_cols = ["N"]
    for f in feature_cols:
        data[f] = get_df_data(df=features_df, col="features", key=f)


    downsampling_threshold = 0.001  # 0.01%
    print(f"Downsampling threshold {downsampling_threshold*100}%")
    data["downsampling"] = data.apply(lambda row: filter_blocks(row), axis=1)

    # Diagonal-like class
    data['offset_density'] = data.apply(
        lambda row: calculate_offset_density(row["k_dist"], row["N"]), axis=1
    )

    # Triangular class
    upper_diag_indices = [i * 10 + j for i in range(10) for j in range(i + 1, 10)]
    lower_diag_indices = [i * 10 + j for i in range(1, 10) for j in range(i)]

    data['upper_percentage'] = data["downsampling"].apply(
        lambda x: len([x[idx] for idx in upper_diag_indices if x[idx] > 0])
        / len(upper_diag_indices)
    )
    data['lower_percentage'] = data["downsampling"].apply(
        lambda x: len([x[idx] for idx in lower_diag_indices if x[idx] > 0])
        / len(lower_diag_indices)
    )

    # row col dense
    data['row_spread'] = data.apply(lambda row: get_row_spread(row['blocks_10'], N=row['N']), axis=1)
    data['col_spread'] = data.apply(lambda row: get_col_spread(row['blocks_10'], N=row['N']), axis=1)
    data['is_col_dense'] = (data['col_spread'].apply(lambda x: get_long_spread(x))) & (data['col_spread'].apply(lambda x: has_big_shift(x))) 
    data['is_row_dense'] = (data['row_spread'].apply(lambda x: get_long_spread(x))) & (data['row_spread'].apply(lambda x: has_big_shift(x))) 


    # dispersed Class
    data['disperseness'] = data["downsampling"].apply(lambda x: round(sum(x) / 100.0, 2))

    # packing
    df = pd.DataFrame(index=data.index)
    cols = ['offset_density', 'upper_percentage', 'lower_percentage', 'row_spread', 'col_spread', 'is_col_dense', 'is_row_dense', 'disperseness']
    df["class_features"]= data[cols].apply(lambda row: row.to_dict(), axis=1)

    print (f"Saving {output_file}...")
    df.to_pickle(output_file)
