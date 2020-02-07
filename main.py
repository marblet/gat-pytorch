from gat import create_gat_model
from train import run
from data import load_data

if __name__=='__main__':
    data = load_data('cora')
    model, optimizer = create_gat_model(data)
    run(data, model, optimizer, niter=10, verbose=True)
