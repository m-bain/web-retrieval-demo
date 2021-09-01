import argparse
from client_server import ClientIndex


def process_text_query(
        text_query: str,
        topk: int = 5,
        machine_name: str = 'localhost',
        text_port: int = 12011,
        index_ports: list = [12010],
):
    machine_ports_index = [(machine_name, x) for x in index_ports]
    machine_port_text = (machine_name, text_port)

    client_index = ClientIndex(machine_ports_index, machine_port_text)
    D, I, urls = client_index.search([text_query], topk)
    D, I, urls = D[::-1], I[::-1], urls[::-1]
    print(D, I, urls)
    return [
        {"videoid": str(vidx),
         "similarity": "Similarity: %.2f" % dist,   # reduce size
         "relurl": url
         }
        for vidx, dist, url in zip(I, D, urls)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### faiss hparams
    parser.add_argument('--topk', default=5, type=int,
                        help='Topk search results for queries')
    parser.add_argument('--text_query', default='beautiful woman dancing', type=str,
                        help='text query to search for')
    parser.add_argument('--text_port', default=12011, type=int,
                        help='port number to list for text encoding')
    parser.add_argument('--index_ports', default=[12010], type=list,
                        help='Machine ports to listen to')
    parser.add_argument('--machine_name', default='localhost', type=str,
                        help='Machine name')
    args = parser.parse_args()

    res = process_text_query(
        args.text_query,
        args.topk,
        args.machine_name,
        args.text_port,
        args.index_ports
    )
    print(res)
