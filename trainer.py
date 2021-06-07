import torch
import torch.utils.data




class Optimiser():
    def __init__(self, args):
        self.lrate_sgd = args.lrate_sgd 
        self.lrate_hebb = args.lrate_hebb
        self.hebb_normaliser = args.hebb_normaliser 
        self.perform_sgd = args.perform_sgd 
        self.perform_hebb = args.perform_hebb 
        self.gating = args.gating
        

    def step(self,model,x_batch, r_batch):
        
        if self.perform_sgd:
            self.sgd_update(model,x_batch,r_batch)
        if self.perform_hebb:
            if self.gating=='SLA':
                self.sla_update(model,x_batch)
            elif self.gating=='GHA':
                self.gha_update(model,x_batch)
        


    def sgd_update(self,model,x_batch,r_batch):
        '''
        performs sgd update 
        '''
        y_ = model(x_batch)
        # compute loss 
        loss = model.loss_funct(r_batch, y_)
        # get gradients 
        loss.backward()
        # update weights 
        with torch.no_grad():
            for theta in model.parameters():
                theta -= theta.grad*self.lrate_sgd
            model.zero_grad()

    def sla_update(self,model,x_batch):
        '''
        performs update with subspace learning algorithm
        '''
        x_batch = torch.t(x_batch[0,:]) # 27x1
        with torch.no_grad():
            Y = torch.t(model.W_h) @ x_batch 
            model.W_h += torch.t((torch.outer(Y,x_batch) - torch.outer(Y,model.W_h @ Y) / self.hebb_normaliser) * self.lrate_hebb)
            model.zero_grad()

    def gha_update(self, model, x_batch):
        '''
        performs update with generalised hebbian algorithm
        '''
        x_batch = torch.t(x_batch[0,:]) # 27x1
        with torch.no_grad():
            Y = torch.t(model.W_h) @ x_batch            
            model.W_h += torch.t((torch.outer(Y,x_batch) - (torch.tril(torch.outer(Y,Y)) @ torch.t(model.W_h)) / self.hebb_normaliser) * self.lrate_hebb)
            model.zero_grad()

    # def tril(self,X):
    #     # get lower triangular
    #     X = torch.tril(X)
    #     # flatten
    #     X = X.reshape(1, -1)
    #     X = X.squeeze()
    #     return X