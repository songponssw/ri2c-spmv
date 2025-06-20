# Process format_thead (ft)
# create a column coo_0 which has metrics as a dictionary
from utils import *
import re

kind_mapping = {
    "2d/3d problem": "2d or 3d problem",
    "random 2d/3d problem": "2d or 3d problem",
    "subsequent 2d/3d problem": "2d or 3d problem",
    "2d/3d problem sequence": "2d or 3d problem",
    "acoustics problem": "acoustics problem",
    "chemical oceanography problem": "chemical problem",
    "chemical process simulation problem": "chemical problem",
    "chemical process simulation problem sequence": "chemical problem",
    "computational chemistry problem": "chemical problem",
    "circuit simulation matrix": "circuit simulation problem",
    "circuit simulation problem": "circuit simulation problem",
    "circuit simulation problem sequence": "circuit simulation problem",
    "subsequent circuit simulation problem": "circuit simulation problem",
    "frequency domain circuit simulation problem": "circuit simulation problem",
    "combinatorial problem": "combinatorial problem",
    "computational fluid dynamics": "computational fluid dynamics problem",
    "computational fluid dynamics problem": "computational fluid dynamics problem",
    "computational fluid dynamics problem sequence": "computational fluid dynamics problem",
    "subsequent computational fluid dynamics problem": "computational fluid dynamics problem",
    "computer graphics/vision problem": "computer graphics or vision problem",
    "computer vision problem": "computer graphics or vision problem",
    "counter example problem": "counter example problem",
    "data analytics problem": "data analytics problem",
    "directed graph": "directed graph problem",
    "directed multigraph": "directed graph problem",
    "directed temporal multigraph": "directed graph problem",
    "directed weighted graph": "directed graph problem",
    "directed weighted graph sequence": "directed graph problem",
    "directed weighted random graph": "directed graph problem",
    "directed weighted temporal graph": "directed graph problem",
    "directed weighted temporal multigraph": "directed graph problem",
    "weighted directed graph": "directed graph problem",
    "economic problem": "economic problem",
    "electromagnetics problem": "electromagnetics problem",
    "linear programming problem": "linear programming problem",
    "materials problem": "materials problem",
    "eigenvalue/model reduction problem": "model reduction problem",
    "model reduction problem": "model reduction problem",
    "optimal control problem": "optimal control problem",
    "optimization problem": "optimization problem",
    "optimization problem sequence": "optimization problem",
    "subsequent optimization problem": "optimization problem",
    "power network problem": "power network problem",
    "power network problem sequence": "power network problem",
    "subsequent power network problem": "power network problem",
    "semiconductor device problem": "semiconductor device problem",
    "semiconductor device problem sequence": "semiconductor device problem",
    "semiconductor process problem": "semiconductor device problem",
    "subsequent semiconductor device problem": "semiconductor device problem",
    "statistical/mathematical problem": "statistical or mathematical problem",
    "structural problem": "structural problem",
    "structural problem sequence": "structural problem",
    "subsequent structural problem": "structural problem",
    "theoretical/quantum chemistry problem": "theoretical or quantum chemistry problem",
    "subsequent theoretical/quantum chemistry problem": "theoretical or quantum chemistry problem",
    "theoretical/quantum chemistry problem sequence": "theoretical or quantum chemistry problem",
    "thermal problem": "thermal problem",
    "undirected graph": "undirected graph problem",
    "undirected graph sequence": "undirected graph problem",
    "undirected graph with communities": "undirected graph problem",
    "undirected multigraph": "undirected graph problem",
    "undirected random graph": "undirected graph problem",
    "undirected weighted graph": "undirected graph problem",
    "undirected weighted graph sequence": "undirected graph problem",
    "undirected weighted random graph": "undirected graph problem",
    "weighted undirected graph": "undirected graph problem",
    "random unweighted graph": "undirected graph problem",
    "random undirected graph": "undirected graph problem",
    "random undirected weighted graph": "undirected graph problem",
}


# ROOT_DIR = "/home/thor/incit-spmv"
# WORK_DIR = os.path.join(ROOT_DIR, "analysis-2")

format_list = ["coo", "csr", "dia", "ell"]
thread_list = ["0", "8"]
ft_config = [f"{_ff}_{_tt}" for _tt in thread_list for _ff in format_list]


# def fullpath_to_key(path: str):
#     s = path.replace(".mtx", "")
#     s = s.split("/")
#     group = s[-3]
#     name = s[-2]
#     return f"{group}--{name}"


def create_df(dataset):

    mtx_names_file = os.path.join(ROOT_DIR, f"scripts/mtx-names/{dataset}.name")
    key_list = []
    with open(mtx_names_file) as f:
        for line in f:
            key_list.append(fullpath_to_key(line))

    df = pd.DataFrame(index=key_list)
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


def read_info_from_csv():
    info_file = os.path.join(ROOT_DIR, "output/suitsparse.csv")
    info_df = read_and_clean_csv(info_file)
    if info_df.empty:
        return pd.DataFrame()
    info_df["kind"] = info_df["kind"].map(lambda x: kind_mapping.get(x))

    return info_df[["kind", "group"]].set_index(info_df["key"])


def read_feature_from_csv(dataset: str):
    feature_file = os.path.join(ROOT_DIR, f"output/{dataset}/features.csv")
    feature = read_and_clean_csv(feature_file)
    if feature.empty:
        return pd.DataFrame()
    feature["key"] = feature["mtx"].apply(lambda x: fullpath_to_key(x))
    feature = feature.drop(columns="mtx")
    feature = feature.set_index("key")

    return feature


def update_ft_column(existing_series: pd.Series, new_series: pd.Series) -> pd.Series:
    """Update a column of dictionaries by merging with new values."""
    unpacked = existing_series.apply(pd.Series)
    unpacked.update(new_series)
    return unpacked.apply(lambda row: row.to_dict(), axis=1)


def get_cache_label(wss):
    l3_cache = 16 * 2**20  # (16MiB)
    l2_cache = 2 * 2**20 / 8.0  # (2MiB, 8 inst = 250 KiB)
    l1_cache = 256 * 2**10 / 8.0  # (256 KiB, 8 inst = 32 KiB)
    if wss <= l1_cache:
        return "l1"
    if wss <= l2_cache:
        return "l2"
    if wss <= l3_cache:
        return "l3"
    return "not fit"


def get_derive_features(ff_df: pd.DataFrame):
    ret = ff_df
    # Matrix relate features
    ret["density"] = ret["nnz"] / (ret["N"] * ret["N"])
    ret["skew_nnz"] = (ret["max_nnz_row"] - ret["avg_nnz_row"]) / (ret["avg_nnz_row"])

    # Memory feature
    int_byte = 4
    double_byte = 8

    ret["coo_mem"] = (
        (2 * double_byte * ret["N"])
        + (2 * int_byte * ret["nnz"])
        + (double_byte * ret["nnz"])
    )

    ret["coo_mem"] = (
        (2 * double_byte * ret["N"])
        + (2 * int_byte * ret["nnz"])
        + (double_byte * ret["nnz"])
    )

    ret["csr_mem"] = (int_byte * (ret["nnz"] + ret["N"] + 1)) + (
        double_byte * (ret["nnz"])
    )
    ret["dia_mem"] = (int_byte * ret["num_diags"]) + (
        double_byte * (ret["num_diags"] * ret["stride"])
    )
    ret["ell_mem"] = 12 * ret["N"] * ret["max_nnz_row"]

    # cache label
    for fm in format_list:
        ret[f"{fm}_cache-label"] = ret[f"{fm}_mem"].apply(get_cache_label)

    return ret


if __name__ == "__main__":
    print("Preprocess features from csv file and kind/groups\n")
    dataset = sys.argv[1]
    output_file = os.path.join(WORK_DIR, f"pkl/{dataset}-features.pkl")

    df = create_df(dataset)

    feature_df = read_feature_from_csv(dataset)
    feature_df = get_derive_features(feature_df)
    info_df = read_info_from_csv()

    df["features"] = feature_df.apply(lambda x: x.to_dict(), axis=1)
    df["info"] = info_df.apply(lambda x: x.to_dict(), axis=1)

    df.to_pickle(output_file)
