import numpy as np
import pandas as pd
from pathlib import Path
import faiss
from faiss.contrib.client_server import run_index_server
import argparse


def create_index(
        index: str,
        embed_dim: int,
        nlist: int,
):

    index = faiss.index_factory(embed_dim, f"{index}{nlist},Flat", faiss.METRIC_INNER_PRODUCT)
    return index


def load_feats(
        feat_fp: Path,
        normalize: bool = True
):
    feats = np.load(feat_fp)
    if normalize:
        faiss.normalize_L2(feats)
    return feats


def main(args):

    db_dir = args.data_root / args.frozen_model / 'indexes' / f'WebVid_{args.cut}'
    # check if index already exists:
    # not saving nlist value, just for simplicity's sake...
    # index_fn = f"{args.index}_{nlist}nlist_{args.split}-db_{args.split_to_train_on}-train.index"
    index_fn = f"{args.index}_{args.split}-db_{args.split_to_train_on}-train.index"
    index_fp = str(db_dir / index_fn)
    if args.refresh or not Path(index_fp).is_file():
        db_csv = args.data_root / 'metadata' / f"results_{args.cut}_{args.split}.csv"
        if args.nlist is None:
            dbf = pd.read_csv(db_csv).set_index('videoid')
            # use default nlist = 4 * sqrt(db_len)
            db_len = len(dbf)
            nlist = int(4 * (db_len ** 0.5))
        else:
            nlist = args.nlist

        ## create, train, write index
        print(f"Creating index: {args.index}...")
        index = create_index(args.index, args.embed_dim, nlist)
        if not index.is_trained:
            feats_train_fn = f"vid_embeds_{args.split_to_train_on}.npy"
            print(f"Training index on {feats_train_fn}...")
            feats_train_fp = db_dir / feats_train_fn
            feats_train = load_feats(feats_train_fp, normalize=True)
            index.train(feats_train)

        # add to...
        feats_db_fn = f"vid_embeds_{args.split}.npy"
        ids_db_fn = f"ids_{args.split}.csv"

        print(f"Adding {feats_db_fn} to index...")
        feats_db_fp = db_dir / feats_db_fn
        ids_db_fp = db_dir / ids_db_fn

        feats_db = load_feats(feats_db_fp, normalize=True)
        ids_db = pd.read_csv(ids_db_fp)
        ids_db['videoid'] = ids_db['0'].str.split('/').str[1].str.split('.').str[0]
        ids_db.set_index('videoid', inplace=True)
        ids_db.index = ids_db.index.astype(int)
        ids = ids_db.index.values

        index.add_with_ids(feats_db, ids)
        print(f"... added database of size {index.ntotal} to index.")
        print(f"Writing index to ...\n{index_fn}")

        faiss.write_index(index, index_fp)


    machine_ports = [
        (args.machine, args.port_num),
    ]
    v6 = False
    print(f"Reading index from...\n{index_fp}")
    index = faiss.read_index(index_fp)

    port = machine_ports[0][1]
    print(f"Running server at port {port}...")
    run_index_server(index, port, v6=v6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### faiss hparams
    parser.add_argument('--nlist', default=None, type=int,
                        help='Number of centroids to use')
    parser.add_argument('--embed_dim', default=256, type=int,
                        help='dimension size of embeddings')
    parser.add_argument('--index', default='IVF', choices=['IVF'],
                        help='Choice of FAISS Index.')
    parser.add_argument('--refresh', action='store_true',
                        help='Retrain and resave index')
    ### database params
    parser.add_argument('--frozen_model', default='CC-WebVid2M-4f-pt1f/0522_143949', type=Path,
                        help='Chosen frozen_in_time model to use')
    parser.add_argument('--data_root', default='./data', type=Path,
                        help='Directory where embeds are saved to.')
    parser.add_argument('--cut', default='2M', type=str)
    parser.add_argument('--split', default='test', type=str)
    parser.add_argument('--split_to_train_on', default='test', type=str)
    parser.add_argument('--n_partitions', default=1, type=int)
    parser.add_argument('--part', default=0, type=int)
    parser.add_argument('--port_num', default=12010, type=int,
                        help='port number')
    parser.add_argument('--machine', default='localhost', type=str,
                        help='machine name')
    args = parser.parse_args()

    main(args)