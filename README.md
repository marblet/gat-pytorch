# gcn-pytorch
This is the pytorch inplementation of Graph Attention Networks.

Petar Veličković et al.[Graph Attention Networks](https://arxiv.org/abs/1710.10903)

## Usage
```
$ python main.py
```

## Explanation of main.py
```python
from gat import create_gat_model
from train import run
from data import load_data

if __name__=='__main__':
    # load a data according to input
    data = load_data('cora')

    # create GCN and optimizer(Adam)
    model, optimizer = create_gat_model(data)

    # run the model niter times
    run(data, model, optimizer, niter=10)
```
