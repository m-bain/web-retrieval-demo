import numpy as np
import pandas as pd
from pathlib import Path
import faiss
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from faiss.contrib.client_server import run_index_server, ClientIndex


### load all the stuff
def load_db(data_root, model, csv_fn):
    ftrs_root = data_root / 'features'

    db_feats = np.load(ftrs_root / model / 'results_2M' / 'embeds_test.npy')
    videoids = pd.read_csv(ftrs_root / model / 'results_2M' / 'ids_test.csv')
    df = pd.read_csv(data_root / 'metadata' / csv_fn)
    videoids['videoid'] = videoids['0'].str.split('/').str[1].str.split('.').str[0]
    df = df[df['videoid'].isin(videoids['videoid'])].set_index('videoid')
    videoids.set_index('videoid', inplace=True)
    videoids.index = videoids.index.astype(int)
    df.index = df.index.astype(int)

    videoids['name'] = df['name']

    return db_feats, df, videoids


def main():
    data_root = Path('/scratch/shared/beegfs/maxbain/datasets/ShutterVids')
    model = Path('CC-WebVid2M-4f-pt1f/0522_143949')
    csv_fn = 'results_2M_test.csv'
    ### set up the index
    d = 256
    nlist = 100
    k = 5

    server_query = True

    text_arch = 'distilbert-base-uncased'
    checkpoint_fp = '/users/maxbain/Libs/frozen-in-time/exps/models/CC-WebVid2M-4f-pt1f/0522_143949/checkpoint-epoch6_statedict.pth'
    queries = ['beautiful woman dancing', 'aerial mountain view', 'man running down the street']
    queries = [queries[0]]

    text_model = AutoModel.from_pretrained(text_arch)
    chkpt_state = torch.load(checkpoint_fp)
    tokenizer = AutoTokenizer.from_pretrained(text_arch)
    txt_proj = nn.Sequential(nn.ReLU(),
                             nn.Linear(text_model.config.hidden_size, d),
                             )
    txt_statedict = {}
    txtproj_statedict = {}
    for key, val in chkpt_state.items():
        if key.startswith('module.text_model.'):
            newkey = key[18:]
            txt_statedict[newkey] = val
        elif key.startswith('module.txt_proj.'):
            newkey = key[16:]
            txtproj_statedict[newkey] = val

    text_model.load_state_dict(txt_statedict)
    txt_proj.load_state_dict(txtproj_statedict)

    text_model = text_model.eval()

    query_tokens = tokenizer(queries, return_tensors='pt', padding=True, truncation=True)
    text_feats = text_model(query_tokens['input_ids'], attention_mask=query_tokens['attention_mask']).last_hidden_state[
                 :, 0, :]
    text_feats = txt_proj(text_feats).cpu().detach().numpy()
    db_feats, df, videoids = load_db(data_root, model, csv_fn)
    if server_query:
        machine_ports = [
            ('localhost', 12010),
        ]
        client_index = ClientIndex(machine_ports)
        print('index size:', client_index.ntotal)
        D, I = client_index.search(text_feats, 5)
        print(D, I)
        import pdb;
        pdb.set_trace()
    else:

        # quantizer = faiss.IndexFlatIP(d)
        # index = faiss.IndexIVFFlat(quantizer, d, nlist)

        index = faiss.index_factory(d, f"IVF{nlist},Flat", faiss.METRIC_INNER_PRODUCT)
        print(index.is_trained)

        # normalise
        faiss.normalize_L2(db_feats)
        faiss.normalize_L2(text_feats)

        # train
        print('training...')
        index.train(db_feats)
        print('... done')
        print(index.is_trained)
        index.add_with_ids(db_feats, df.index.values)  # add vec    tors to the index
        print(index.ntotal)

        D, I = index.search(text_feats, k)
        print(D, I)
        for idx, quer in enumerate(queries):
            topknn = I[idx]
            similarities = pd.Series(D[idx])
            similarities.index = topknn
            tdf = videoids.loc[topknn]
            tdf['sim_score'] = similarities
            # tdf.sort_values('sim_score', inplace=True, ascending=False)
            print(f"#### query: {quer} ####\n", tdf[['name', 'sim_score']])


main()
