from utils import *

if __name__ == "__main__":
    dataset = sys.argv[1]
    filenames = ["features", "perf-metrics", "submatrices", "class_features", "class-2"]
    output_file = os.path.join(WORK_DIR, "pkl", f"{dataset}-merged.pkl")

    # kdensity
    # require k_dist
    print("Join everything in to data")
    data = pd.DataFrame()
    for filename in filenames:
        filepath = os.path.join(WORK_DIR, "pkl", f"{dataset}-{filename}.pkl")
        tmp_df = pd.read_pickle(filepath)
        if data.empty:
            data = tmp_df
            continue
        data = data.join(tmp_df)

    print (f"Saving {output_file}...")
    data.to_pickle(output_file)

