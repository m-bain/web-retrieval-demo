import pandas as pd
from pathlib import Path
from client_server import run_text_server
from transformers import AutoModel, AutoTokenizer
import argparse
import json
import torch
import torch.nn as nn


class FrozenTextEncoder(nn.Module):
    def __init__(self, text_model, projection):
        super().__init__()
        self.text_model = text_model
        self.projection = projection

    def forward(self, x):
        feats = self.text_model(x['input_ids'],
                                attention_mask=x['attention_mask']).last_hidden_state[:, 0, :]
        feats = self.projection(feats)
        return feats


def main(args):
    config_fp = args.model_dir / args.frozen_model / 'config.json'
    with open(config_fp, 'r') as fid:
        config = json.load(fid)
    text_arch = config['arch']['args']['text_params']['model']
    projection = config['arch']['args'].get('projection', 'minimal')
    projection_dim = config['arch']['args'].get('projection_dim', 256)

    text_model = AutoModel.from_pretrained(text_arch)
    tokenizer = AutoTokenizer.from_pretrained(text_arch)

    if projection == 'minimal':
        txt_proj = nn.Sequential(nn.ReLU(),
                                 nn.Linear(text_model.config.hidden_size, projection_dim),
                                 )
    else:
        raise NotImplementedError

    checkpoint_fp = args.model_dir / args.frozen_model / args.statedict_fn
    chkpt_state = torch.load(checkpoint_fp, map_location=torch.device('cpu'))

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
    text_model = FrozenTextEncoder(text_model, txt_proj)

    # url data
    db_csv = args.data_root / args.frozen_model / 'indexes' / f"{args.dataset}_{args.cut}" / f"{args.split}.csv"
    dbf = pd.read_csv(db_csv).set_index('videoid')
    dbf['relUrl'] = dbf['contentUrl'].str[41:]
    url_data = dbf['relUrl']
    machine_port = (args.machine, args.port_num)
    print(f"Running server for {text_arch}, listening on port {machine_port}")
    run_text_server(tokenizer, text_model, url_data, machine_port[1], v6=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen_model', default='CC-WebVid2M-4f-pt1f/0522_143949', type=Path,
                        help='Chosen frozen_in_time model to use')
    parser.add_argument('--statedict_fn', default='checkpoint-epoch6_txt_statedict.pth', type=str,
                        help='Chosen checkpoint state_dict')
    parser.add_argument('--model_dir', default='./data', type=Path,
                        help='Directory where models are saved to.')
    parser.add_argument('--data_root', default='./data', type=Path,
                        help='Directory where embeds are saved to.')
    parser.add_argument('--dataset', default='WebVid', type=str)
    parser.add_argument('--cut', default='2M', type=str)
    parser.add_argument('--split', default='test', type=str)
    parser.add_argument('--port_num', default=12011, type=int,
                        help='port number')
    parser.add_argument('--machine', default='localhost', type=str,
                        help='machine name')
    args = parser.parse_args()

    main(args)
