import pandas as pd
pd.options.mode.chained_assignment = None 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import MSELoss, L1Loss
from sklearn.model_selection import train_test_split
from utils import *

class dnn(object):
    
    def __init__(self, X, y, methodology, n_hidden = 2, n_nodes = 30, iters = 2000, q=None, M_super=None, side=None):
        self.X = X
        self.y = y
        self.n_hidden = n_hidden
        self.n_nodes = n_nodes
        self.iters = iters
        self.metho = methodology
        self.M = M_super
        self.side = side
        if self.metho == 'quantile':
            self.q = q
        elif self.metho == 'superquantile':
            self.q = q
            if self.side == 'right':
                delta = (1-self.q) / self.M
                w = delta / (1-self.q)
                self.quants = [ self.q + (j-0.5)*delta for j in range(1,self.M+1) ]
            else:
                delta = (self.q) / self.M
                w = delta / (self.q)
                self.quants = [ self.q - (j-0.5)*delta for j in range(1,self.M+1) ]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)
    
    def train(self):
        output_feature = "y"
        self.X_train['y'] = self.y_train.values
        self.X_test['y'] = self.y_test.values
        train_ds = TabularDataset(data=self.X_train, output_col=output_feature)
        test_ds = TabularDataset(data=self.X_test, output_col=output_feature)
        batchsize = 64
        train_dl = DataLoader(train_ds, batch_size = batchsize, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=len(test_ds))
        lr = 1e-3
        seed = 0
        torch.cuda.empty_cache()
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        criterion_mae = L1Loss()
        
        if self.metho != 'point':
            if self.metho == 'quantile':
                quantiles = [self.q]
            else:
                quantiles = self.quants
            model = FeedForwardNN(no_of_cont=self.X.shape[1], lin_layer_sizes=[self.n_nodes for i in range(self.n_hidden)], output_size=len(quantiles),
                                        lin_layer_dropouts=[0.05 for i in range(self.n_hidden)]).to(device)
            model.apply(weights_init)
            criterion = QuantileLoss(quantiles=sorted(quantiles))
            optimizer = optim.Adam(model.parameters(), lr = lr)
        else:
            model = FeedForwardNN(no_of_cont=self.X.shape[1], lin_layer_sizes=[self.n_nodes for i in range(self.n_hidden)], output_size=1,
                                        lin_layer_dropouts=[0.05 for i in range(self.n_hidden)]).to(device)
            model.apply(weights_init)
            criterion = MSELoss()
            optimizer = optim.Adam(model.parameters(), lr = lr)
        
        best_loss = 100
        for epoch in range(self.iters):
            ###### Training ######
            model = model.train()
            for y, cont_x in train_dl:
                optimizer.zero_grad()
                cont_x = cont_x.to(device)
                y  = y.to(device)
                output_t = model(cont_x)
                loss = criterion(output_t, y)
                loss.backward()
                optimizer.step()
            
            ###### Validation ######
            model = model.eval()
            test_loss = 0
            with torch.no_grad():
                for y, cont_x in test_dl:
                    cont_x = cont_x.to(device)
                    y  = y.to(device)
                    output_test = model(cont_x)
                    loss = criterion(output_test, y) 
                    loss_mae = criterion_mae(output_test, y) 
                    test_loss += loss.item()
            
            test_loss /= float(len(test_dl))
            if test_loss < best_loss:
                torch.save(model.state_dict(), 'best_model_q.pt')
                best_loss = test_loss
                best_mae = loss_mae
                last_save = epoch
        
        model.load_state_dict(torch.load('best_model_q.pt'))
        
        if self.metho == 'point':
            best_loss = best_mae
        
        print('NN fitting process finished, with a test MAE/Qloss of', best_loss, 'in epoch', last_save)
        return model