import argparse
import lmdb
import pickle
from pathlib import Path
from tqdm import tqdm


def save_to_lmdb(pkl_path, lmdb_path):
    lmdb_path.mkdir(exist_ok=True, parents=True)
    db = lmdb.open(
        str(lmdb_path / "data.lmdb"),
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    total_len = 0
    all_graphs = list((pkl_path).glob("*.pkl"))
    for idx, pkl_file in enumerate(tqdm(all_graphs)):
        with open(str(pkl_file), "rb") as f:
            data = pickle.load(f)
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()
        total_len += 1

    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(total_len, protocol=-1))
    txn.commit()
    db.sync()
    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkl_path",
        type=str,
        help="path to processed graphs",
    )
    parser.add_argument(
        "--lmdb_path",
        type=str,
        help="output path for lmdb file",
    )
    args = parser.parse_args()
    save_to_lmdb(Path(args.pkl_path), Path(args.lmdb_path))
