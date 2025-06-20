# The process_class_features produces
# - submatrices 10x10.
# - class features
from utils import *
from collections import defaultdict
import scipy.sparse as sp
from scipy.io import mmwrite
import scipy

def downsampling_coo_by_blocks(matrix, num_blocks):
    M, N = matrix.shape

    block_size_row = M // num_blocks
    block_size_col = N // num_blocks

    # Compute block indices
    row_bins = np.minimum(
        matrix.row // block_size_row, num_blocks - 1
    )  # Assign overflow rows to last block
    col_bins = np.minimum(
        matrix.col // block_size_col, num_blocks - 1
    )  # Assign overflow cols to last block

    block_dict = defaultdict(float)
    for r, c, v in zip(row_bins, col_bins, matrix.data):
        block_dict[(r, c)] += v

    rows, cols, data = zip(*[(r, c, v) for (r, c), v in block_dict.items()])
    downsampled = sp.coo_matrix((data, (rows, cols)), shape=(num_blocks, num_blocks))

    del rows, cols, data, row_bins, col_bins, block_dict
    gc.collect()

    return downsampled


def key_to_spmatrix(fig_key):
    fig_group, fig_name = fig_key.split("--")
    mtx_filepath = os.path.join(DATASET_DIR, fig_group, fig_name, f"{fig_name}.mtx")
    coo = scipy.io.mmread(mtx_filepath)
    coo.eliminate_zeros()
    return coo


def fullpath_to_key(path: str):
    s = path.replace(".mtx", "")
    s = s.split("/")
    group = s[-3]
    name = s[-2]
    return f"{group}--{name}"


if __name__ == "__main__":

    dataset = sys.argv[1]
    output_file = os.path.join(WORK_DIR, "pkl", f"{dataset}-submatrices.pkl")
    num_blocks = 10

    mtx_names_file = os.path.join(ROOT_DIR, f"scripts/mtx-names/{dataset}.name")
    key_list = []
    with open(mtx_names_file) as f:
        for line in f:
            key_list.append(fullpath_to_key(line))

    data = {}
    for i, idx in enumerate(key_list):
        print(f"{i}: Calculate sub matrices {idx}...")
        coo = key_to_spmatrix(idx)
        coo.data = np.ones_like(coo.data)  # Convert to binary presence (nnz count)
        block_matrix = downsampling_coo_by_blocks(coo, num_blocks)
        data[idx] = block_matrix.todense().A1.tolist()

    df = pd.DataFrame(list(data.items()), columns=["key", "blocks_10"]).set_index("key")
    df.to_pickle(output_file)
