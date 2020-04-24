from gat import create_gat_model
from spgat import create_spgat_model
from train import run
from data import load_data

if __name__=='__main__':
    # load a data according to input
    data = load_data('cora')

    # create GAT model
    # You can use the sparse version of GAT, which reduces computational time and memory consumption.
    model = create_spgat_model(data)
    # You can also use the dense version of GAT
    # model = create_gat_model(data)

    # run the model niter times
    run(data, model, lr=0.005, weight_decay=5e-4, niter=10)
