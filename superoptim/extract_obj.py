import os
import numpy as np
from superdec.utils.predictions_handler import PredictionHandler


def main():
    _in = "data/output_npz/shapenet_test.npz"
    out = "data/output_npz/objects/round_table6.npz"
    idx = 1780

    data = np.load(_in, allow_pickle=True)
    filtered_data = {key: data[key][idx:idx+1] for key in data.files}

    # Save the new subset
    np.savez(out, **filtered_data)
    print(f"Saved index {idx} to {out}")
if __name__ == "__main__":
    main()