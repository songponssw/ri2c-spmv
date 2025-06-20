from utils import *
import ast


def get_percentage_upper_k(l_dist, k_index, nnz):
    nnz_upper = np.array([v for k, v in l_dist.items() if k > k_index])
    return round(sum(nnz_upper) / nnz * 100.0, 2) if nnz != 0 else 0.0


def get_percentage_lower_k(l_dist, k_index, nnz):
    nnz_lower = np.array([v for k, v in l_dist.items() if k < k_index])
    return round(sum(nnz_lower) / nnz * 100.0, 2) if nnz != 0 else 0.0


def percentage_to_k(N, percentage):
    max_k = N - 1
    return max_k * (percentage / 100)

def get_series(fname, col_list):
    ret = []
    raw_df = read_and_clean_csv(fname)
    raw_df["key"] = raw_df["mtx"].apply(lambda x: fullpath_to_key(x))
    raw_df = raw_df.drop(columns="mtx")
    raw_df = raw_df.set_index("key")

    if len(col_list) == 1:
        ret = raw_df[col_list[0]].apply(lambda x: ast.literal_eval(x))
    
    ret = {c: raw_df[c].apply(lambda x: ast.literal_eval(x)) for c in col_list}
    return pd.DataFrame(ret)

if __name__ == "__main__":
    dataset = sys.argv[1]
    # kdist_file = os.path.join(ROOT_DIR, f"analysis/raw/row_dist/{dataset}/k_dist.csv")
    kdist_file = os.path.join(ROOT_DIR, f"output/{dataset}/k_dist.csv")
    print("Processing ...")
    s = get_series(kdist_file, ["k_dist"])
    # s = s.to_frame()
    output_file = os.path.join(WORK_DIR, "pkl", f"{dataset}-k_dist.pkl")
    print(f"Saving {output_file} ...")
    s.to_pickle(output_file)
