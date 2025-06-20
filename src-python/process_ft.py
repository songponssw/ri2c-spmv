# Process format_thead (ft)
# create a column coo_0 which has metrics as a dictionary
from utils import * 
import re


# format_list = ["coo", "csr", "dia", "ell", "csrnoxmiss"]
format_list = ["coo", "csr", "dia", "ell"]
thread_list = ["0", "8"]
ft_config = [f"{_ff}_{_tt}" for _tt in thread_list for _ff in format_list]



def create_df(dataset):
    column_list = ft_config
    metrics = ["exectime", "ldist"]

    mtx_names_file = os.path.join(ROOT_DIR, f"scripts/mtx-names/{dataset}.name")
    key_list = []
    with open(mtx_names_file) as f:
        for line in f:
            key_list.append(fullpath_to_key(line))

    df = pd.DataFrame(columns=column_list, index=key_list)
    df[ft_config] = df[ft_config].map(
        lambda _: {m: [0] if m == "ldist" else 0.0 for m in metrics}
    )
    return df


# def read_and_clean_csv(raw_file):
#     try:
#         ret = pd.read_csv(raw_file)
#         ret = ret.dropna().reset_index(drop=True)
#         ret = ret.drop_duplicates().reset_index(drop=True)
#     except FileNotFoundError:
#         print(f"Not found : {raw_file}")
#         ret = pd.DataFrame()
#     return ret


def get_exectime_from_csv(tar_fm: str, tar_th: str, dataset: str):
    if tar_th == "0":
        bin_file = "exectime"
    else:
        bin_file = "exectime_omp"

    time_dir = os.path.join(ROOT_DIR, f"output/{dataset}")
    raw_file = os.path.join(time_dir, f"run_{bin_file}-{tar_th}-{tar_fm}.csv")

    execute_df = read_and_clean_csv(raw_file)
    if execute_df.empty:
        return pd.DataFrame()

    time_cols = [c for c in list(execute_df.columns) if re.search("outer[0-9]", c)]
    for tc in time_cols:
        execute_df[tc] = pd.to_numeric(execute_df[tc])
        execute_df[tc] = execute_df[tc].div(execute_df["inner"])

    execute_df["rawtime"] = execute_df[time_cols].apply(
        lambda x: pd.to_numeric(x).to_list(), axis=1
    )
    # Avg time in usec
    execute_df["avgtime"] = execute_df["rawtime"].apply(
        lambda x: np.round(np.mean(x), 5)
    )
    execute_df["key"] = execute_df["mtx"].apply(lambda x: fullpath_to_key(x))

    ret = pd.DataFrame({"exectime": execute_df["avgtime"]}).set_index(execute_df["key"])
    return ret


def get_ldist_from_csv(tar_fm: str, tar_th: str, dataset: str):
    bin_file = "ldist"
    time_dir = os.path.join(ROOT_DIR, f"output/{dataset}")
    raw_file = os.path.join(time_dir, f"run_{bin_file}-{tar_th}-{tar_fm}.csv")

    _val_cols = [f"th{i}" for i in range(1, int(tar_th) + 1)]

    execute_df = read_and_clean_csv(raw_file)
    if execute_df.empty:
        return pd.DataFrame()

    execute_df["key"] = execute_df["mtx"].apply(lambda x: fullpath_to_key(x))
    execute_df["val"] = execute_df[_val_cols].apply(lambda x: x.tolist(), axis=1)

    ret = pd.DataFrame({"ldist": execute_df["val"]}).set_index(execute_df["key"])
    return ret


def update_ft_column(existing_series: pd.Series, new_series: pd.Series) -> pd.Series:
    """Update a column of dictionaries by merging with new values."""
    unpacked = existing_series.apply(pd.Series)
    unpacked.update(new_series)
    return unpacked.apply(lambda row: row.to_dict(), axis=1)


if __name__ == "__main__":
    print("Preprocess values related to format-thread (exectime, ldist)\n")
    dataset = sys.argv[1]
    output_file = os.path.join(WORK_DIR, f"pkl/{dataset}-time.pkl")

    df = create_df(dataset)

    # Execution time
    for _ff in format_list:
        for _tt in thread_list:
            _ft = f"{_ff}_{_tt}"
            df[_ft] = update_ft_column(
                existing_series=df[_ft],
                new_series=get_exectime_from_csv(_ff, _tt, dataset),
            )

            # Update ldist
            if _tt == "0":
                continue
            df[_ft] = update_ft_column(
                existing_series=df[_ft],
                new_series=get_ldist_from_csv(_ff, _tt, dataset),
            )

    print (f"Saving: {output_file}")
    df.to_pickle(output_file)
