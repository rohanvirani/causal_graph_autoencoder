import numpy as np
import torch
from torch import optim

from synthetic_dataset import SyntheticDataset
from gae import GAE_torch
from config_utils import save_yaml_config, get_args
from log_helper import LogHelper
from tf_utils import set_seed
from dir_utils import make_new_folder
from analyze_utils import count_accuracy, plot_recovered_graph, save_input_args
from data import CELEBA_LABELS

def prep_data(data, useCUDA):
    y = data
    if useCUDA:
        y = y.cuda().view(y.size(0), -1)
    else:
        y = y.view(y.size(0), -1)
    return y

if __name__ == '__main__':
    opts = get_args()

    #exDir = make_new_folder(opts.output_dir)
    
    #save_input_args(exDir, opts)
        
    dataset = SyntheticDataset(opts.n, opts.d, opts.graph_type, opts.degree, opts.sem_type,
                               opts.noise_scale, opts.dataset_type, opts.x_dim)

    trainDataset = CELEBA_LABELS(root=opts.root, train=True, label=opts.features)
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=opts.batchSize, shuffle=True)

    gae = GAE_torch(d = opts.d, hidden_size=opts.hidden_size, l1_graph_penalty=opts.l1_graph_penalty)
    if gae.useCUDA:
        print('using CUDA')
        gae.cuda()
    else:
        print('\n *** NOT USING CUDA ***\n')

    print(gae)

    optimizerGAE = optim.Adam(gae.parameters(), lr=opts.learning_rate)  # specify the params that are being upated

    losses = {'total': [], 'mse': [], 'h': []}
    Ns = len(trainLoader) * opts.batchSize  # number samples
    Nb = len(trainLoader)  # number batches

    init_rho = opts.init_rho
    rho_thres = opts.rho_thres
    h_thres = opts.h_thres
    rho_multiply = opts.rho_multiply
    init_iter = opts.init_iter
    learning_rate = opts.learning_rate
    h_tol = opts.h_tol
    early_stopping = opts.early_stopping
    early_stopping_thres = opts.early_stopping_thres

    rho, alpha, h, h_new = init_rho, 0.0, np.inf, np.inf 
    prev_W_est, prev_mse = None, float('inf')

    ####### Start Training #######
    for e in range(opts.maxEpochs):
        print('Epoch: ', e)
        gae.train()

        for i, data in enumerate(trainLoader, 0):
            y = prep_data(data, gae.useCUDA)
            output = gae.forward(y.float())
            mse_new, h_new, total_loss = gae.loss(y, output, rho, alpha)
            W_new = gae.adj_A
            
            optimizerGAE.zero_grad()
            total_loss.backward()  # fill in grads
            optimizerGAE.step()

            if i % 100 == 0:
                print(h_new.item())

        if abs(h_new.item()) > h_thres * abs(h):
            rho *= rho_multiply
    
        W_est, h = W_new, h_new
        alpha += rho * h

        print('Rho: ', rho)
        print('Alpha: ',alpha)
        print('MSE: ', mse_new.item())
        print('Total Loss: ',total_loss.item())
        print('h: ',h_new.item())
        print('A: ', gae.adj_A)


        if early_stopping:
            if abs(h_new) <= 0.1:
                # MSE increases too much, revert back to original graph and perform early stopping
                # Only perform this early stopping when h_new is sufficiently small
                # (i.e., at least smaller than 1e-7)
                W_est = prev_W_est
                break
            else:
                prev_W_est = W_new
                prev_mse = mse_new

        if abs(h) <= h_tol and e > init_iter:
            print('Early stopping at {}-th epoch'.format(e))
            break



