# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from multiprocessing.dummy import Pool as ThreadPool
import faiss
from typing import List, Tuple
from faiss.contrib import rpc
from transformers import PreTrainedTokenizer
import torch.nn as nn
import pandas as pd

############################################################
# Server implementation
############################################################


class TextEncodingServer(rpc.Server):
    """ Assign version that can be exposed via RPC """

    def __init__(self, s: int, tokenizer: PreTrainedTokenizer, text_model: nn.Module, url_data: pd.Series):
        rpc.Server.__init__(self, s)
        self.tokenizer = tokenizer
        self.text_model = text_model
        self.url_data = url_data

    def encode(self, text_list):
        tokens = self.tokenizer(text_list, return_tensors='pt', padding=True, truncation=True)
        text_feats = self.text_model(tokens)
        return text_feats.cpu().detach().numpy()

    def get_url(self, indexes, distances):
        if len(indexes) != 1:
            raise ValueError(f"Expecting query batch of size 1, not {len(indexes)}")
        indexes = indexes[0]
        distances = distances[0]

        out_url = []
        out_idx = []
        out_dist = []
        for idx, dist in zip(indexes, distances):
            if idx != -1:
                out_idx.append(idx)
                out_url.append(self.url_data.loc[idx])
                out_dist.append(dist)

        return out_url, out_idx, out_dist

    # def __getattr__(self, f):
    #    # all other functions get forwarded to the index
    #    return getattr(self.index, f)


class SearchServer(rpc.Server):
    """ Assign version that can be exposed via RPC """

    def __init__(self, s: int, index: faiss.Index):
        rpc.Server.__init__(self, s)
        self.index = index
        self.index_ivf = faiss.extract_index_ivf(index)

    def set_nprobe(self, nprobe: int) -> int:
        """ set nprobe field """
        self.index_ivf.nprobe = nprobe

    def get_ntotal(self) -> int:
        return self.index.ntotal

    def __getattr__(self, f):
        # all other functions get forwarded to the index
        return getattr(self.index, f)


def run_index_server(index: faiss.Index, port: int, v6: bool = False):
    """ serve requests for that index forerver """
    rpc.run_server(
        lambda s: SearchServer(s, index),
        port, v6=v6)


def run_text_server(tokenizer: PreTrainedTokenizer,
                    text_model: nn.Module,
                    url_data: pd.Series,
                    port: int,
                    v6: bool = False):
    """ serve requests for that index forerver """
    rpc.run_server(
        lambda s: TextEncodingServer(s, tokenizer, text_model, url_data),
        port, v6=v6)


############################################################
# Client implementation
############################################################

class ClientIndex:
    """manages a set of distance sub-indexes. The sub_indexes search a
    subset of the inverted lists. Searches are merged afterwards
    """

    def __init__(self, machine_ports_search: List[Tuple[str, int]],
                 machine_port_text: Tuple[str, int],
                 v6: bool = False):
        """ connect to a series of (host, port) pairs """
        self.sub_indexes = []
        for machine, port in machine_ports_search:
            self.sub_indexes.append(rpc.Client(machine, port, v6))

        self.text_client = rpc.Client(machine_port_text[0], machine_port_text[1], v6)

        self.ni = len(self.sub_indexes)
        # pool of threads. Each thread manages one sub-index.
        self.pool = ThreadPool(self.ni)
        # test connection...
        self.ntotal = self.get_ntotal()
        self.verbose = False

    def set_nprobe(self, nprobe: int) -> None:
        self.pool.map(
            lambda idx: idx.set_nprobe(nprobe),
            self.sub_indexes
        )

    def set_omp_num_threads(self, nt: int) -> None:
        self.pool.map(
            lambda idx: idx.set_omp_num_threads(nt),
            self.sub_indexes
        )

    def get_ntotal(self) -> None:
        return sum(self.pool.map(
            lambda idx: idx.get_ntotal(),
            self.sub_indexes
        ))

    def search(self, x, k: int):
        """
        :param x: text queries (List[str])
        :param k: topk results returned (int)
        :return: D, I
        """
        x = self.text_client.encode(x)

        rh = faiss.ResultHeap(x.shape[0], k)

        for Di, Ii in self.pool.imap(lambda idx: idx.search(x, k), self.sub_indexes):
            rh.add_result(Di, Ii)
        rh.finalize()

        urls, idx, distances = self.text_client.get_url(rh.I, rh.D)
        return distances, idx, urls
