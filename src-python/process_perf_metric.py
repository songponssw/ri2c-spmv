# process_perf_metric
# - Calculate FLOPS in each format and store in time_df key(mflop, gflop)
# - Create columns ranking to compare speedup 10% split
from utils import *
import re
import gc

format_list = ["coo", "csr", "dia", "ell"]
thread_list = ["0", "8"]
ft_config = [f"{_ff}_{_tt}" for _tt in thread_list for _ff in format_list]



def get_df_data(df, col: str, key: str):
    return pd.DataFrame.from_records(df[col].to_numpy()).set_index(df.index)[key]


def update_ft_column(existing_series: pd.Series, new_series: pd.Series) -> pd.Series:
    """Update a column of dictionaries by merging with new values."""

    unpacked = existing_series.apply(pd.Series)
    new_unpacked = new_series.to_frame()

    # Iterate over rows to update/merge
    updated = unpacked.copy()

    for index, row in new_unpacked.iterrows():
        for key, value in row.dropna().items():  # Ignore NaN values
            if key in updated.columns:
                updated.at[index, key] = value  # Update existing key
            else:
                updated.at[index, key] = value  # Add new key

    return updated.apply(lambda row: row.to_dict(), axis=1)


def split_sp10(row):
    threshold = 1.1
    get_fm = lambda x: x.split("_")[1]
    get_ft = lambda x: x.replace("time_", "")

    row = row.sort_values(ascending=True)
    vv = row.values.tolist()
    if row.values.min() == 0:
        print(f"{ft}\t {row.name}: {row.values.min()}")
    _c = (vv / row.values.min()) >= threshold

    left = row[~_c].index.tolist()
    right = row[_c].index.tolist()

    left = tuple(map(get_ft, left))
    right = tuple(map(get_ft, right))

    return left, right


def get_ranking(time_df):
    tmp = pd.DataFrame()
    print(ft_config)
    for ft in ft_config:
        tmp[f"time_{ft}"] = get_df_data(df=time_df, col=ft, key="exectime")

    ret = pd.DataFrame(index=tmp.index)
    # ranking
    for tt in ["0", "8"]:
        fts = [f"{ff}_{tt}" for ff in format_list]
        left_col = f"left_{tt}"
        right_col = f"right_{tt}"
        ret[[left_col, right_col]] = tmp[[f"time_{_ft}" for _ft in fts]].apply(
            lambda x: pd.Series(split_sp10(x)), axis=1
        )
        ret[f"{left_col}_len"] = ret[left_col].apply(lambda x: len(x))
        ret[f"{right_col}_len"] = ret[right_col].apply(lambda x: len(x))
        ret[f"fastest_fm_{tt}"] = ret.apply(lambda row: f"{row[left_col][0]}", axis=1)

    return ret


def get_sp_and_eff(time_df):
    tmp = pd.DataFrame()
    for ft in ft_config:
        tmp[f"time_{ft}"] = get_df_data(df=time_df, col=ft, key="exectime")

    ret = pd.DataFrame(index=tmp.index)

    # Speedup when scaling
    for ff in format_list:
        ret[f"sp8_{ff}"] = tmp[f"time_{ff}_0"] / tmp[f"time_{ff}_8"]

    # Speedup compare to csr
    for tt in thread_list:
        ret[f"sp{tt}_csr_coo"] = tmp[f"time_csr_{tt}"] / tmp[f"time_coo_{tt}"]
        ret[f"sp{tt}_csr_csr"] = tmp[f"time_csr_{tt}"] / tmp[f"time_csr_{tt}"]
        ret[f"sp{tt}_csr_dia"] = tmp[f"time_csr_{tt}"] / tmp[f"time_dia_{tt}"]
        ret[f"sp{tt}_csr_ell"] = tmp[f"time_csr_{tt}"] / tmp[f"time_ell_{tt}"]

    # Efficiency
    for tt in ["8"]:
        tt = int(tt)
        ret[f"eff{tt}_coo"] = round(ret[f"sp{tt}_coo"] / tt, 4)
        ret[f"eff{tt}_csr"] = round(ret[f"sp{tt}_csr"] / tt, 4)
        ret[f"eff{tt}_dia"] = round(ret[f"sp{tt}_dia"] / tt, 4)
        ret[f"eff{tt}_ell"] = round(ret[f"sp{tt}_ell"] / tt, 4)

    return ret


if __name__ == "__main__":
    print("Preprocess performance metrics")

    dataset = sys.argv[1]
    pkl_dir = os.path.join(WORK_DIR, "pkl")
    time_pkl = os.path.join(pkl_dir, f"{dataset}-time.pkl")
    feature_pkl = os.path.join(pkl_dir, f"{dataset}-features.pkl")
    output_file = os.path.join(WORK_DIR, f"pkl/{dataset}-perf-metrics.pkl")

    time_df = pd.read_pickle(time_pkl).sort_index()
    feature_df = pd.read_pickle(feature_pkl).sort_index()

    # checking both having same index order
    if not time_df.index.equals(feature_df.index):
        print(
            "Error: The time_df and feature_df have different indices.", file=sys.stderr
        )
        sys.exit(1)

    # Calculate FLOPS = 2*nnz/time
    nnz_df = get_df_data(df=feature_df, col="features", key="nnz")
    for ft in ft_config:
        print(f"Calcuating FLOPS on {ft}")
        _series = get_df_data(df=time_df, col=ft, key='exectime')  # usec
        flops = (2 * nnz_df) / _series
        mflops = round(flops, 4)
        gflops = round(flops / 1e3, 4)
        mflops.name = "mflops"
        gflops.name = "gflops"
        time_df[ft] = update_ft_column(existing_series=time_df[ft], new_series=mflops)
        time_df[ft] = update_ft_column(existing_series=time_df[ft], new_series=gflops)

    # Calculate imb
    for ff in format_list:
        ft = f"{ff}_8"
        print(f"Calcuating imbalance ratio on {ft}")
        ldist = get_df_data(df=time_df, col=ft, key="ldist")  # usec
        imb = ldist.apply(np.max) / ldist.apply(np.mean)
        imb = round(imb, 4)
        imb.name = "imb"
        time_df[ft] = update_ft_column(existing_series=time_df[ft], new_series=imb)

    # Calculate Speedup
    sp_df = pd.DataFrame()
    print(f"Calcuating Speedup and Efficiency")
    sp_df["speedup"] = get_sp_and_eff(time_df).apply(lambda x: x.to_dict(), axis=1)

    # Calculate raking split speedup 10%
    print(f"Calcuating ranking")
    rank_df = pd.DataFrame()
    rank_df["rank"] = get_ranking(time_df).apply(lambda x: x.to_dict(), axis=1)

    # Concat
    df = pd.DataFrame()
    df = time_df.join([sp_df, rank_df])
    df.to_pickle(output_file)
