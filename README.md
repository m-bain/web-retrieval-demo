# Demo for Text-to-Video Retrieval

## Preparation

#### 1. environment

    conda env create -f environment.yml

#### 2. Copy data directory containing indexes/model (7GB) to current 
    scp user@triton.robots.ox.ac.uk:/work/maxbain/Libs/web-retrieval-demo/data.zip .
    
    unzip data.zip
    
    rm data.zip

### Run demo

Runs two separate jobs, (one for text encoding, the second for index searching)
1. start index job: `python frozen_server_index.py` (change machine name / port numbers using args, defaults to local 12010)
2. start text encoder job: `python frozen_server_encoder.py`
3. Run server: `export FLASK_APP=demo/demo.py; flask run`

Browse website with local machine (if running on remote host):
`ssh -L 16006:127.0.0.1:5000 user@remotehost`

Visit http://127.0.0.1:16006/


#### Deploy to production?

Standalone Flask is apparently not designed for production https://flask.palletsprojects.com/en/2.0.x/tutorial/deploy/

I am not sure what you typically use in production but the guide uses `Waitress` as an example
>use a production WSGI server. For example, to use Waitress
>$ pip install waitress





