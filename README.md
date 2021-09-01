# Demo for Text-to-Video Retrieval

## Preparation

#### 1. environment

    conda env create -f environment.yml

#### 2. Copy data directory containing indexes/model (7GB) to current 
    scp user@triton.robots.ox.ac.uk:/work/maxbain/Libs/web-retrieval-demo/data.zip .
    
    unzip data.zip
    
    rm data.zip

### Run demo

Runs two separate jobs and a server, (one for text encoding, the second for index searching, please use different sessions / screens for each job and server, so this is 3 sessions/jobs in total)
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


### GENERAL OUTLINE.

1. User writes text query in web form

2. Web form sent to client

3. client sends to text encoder server

4. text encoder server sends back a vector to client

5. client sends vector to index servers

6. server returns topk videos, similarity info etc.

7. Client displays results on web via js script

The implementations of client / server are in `client_server.py`


