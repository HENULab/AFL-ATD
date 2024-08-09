# System

- `main.py`: configurations of **FedATD**. 
- `env_linux.yaml`: python environment to run **FedATD** on Linux. 
- `./flcore`: 
    - `./clients/clientatd.py`: the code on the client. 
    - `./servers/serveratd.py`: the code on the server. 
    - `./trainmodel/models.py`: the code for backbones. 
- `./utils`:
    - `ALA.py`: the code of our Adaptive Local Aggregation module
    - `data_utils.py`: the code to read the dataset. 
# Simulation

## Environments
With the installed [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh), we can run this platform in a conda virtual environment called *fl*. 
```
conda env create -f env_cuda_latest.yaml # for Linux
```


## Training and Evaluation

All codes corresponding to **FedATD** are stored in `./AFL-ATD`. Just run the following commands.

```
pip install -r requtrements.txt
run main.py
```
