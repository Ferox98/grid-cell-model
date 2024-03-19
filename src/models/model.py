from src.hopfield import ModernHopfield, SlotMemory
from src.conf import * 
from src.models.graph import Graph 
from src.utils import * 

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter 
import torchviz



from typing import List 
import matplotlib.pyplot as plt 


class GridModule(nn.Module):
    def __init__(self):
        super(GridModule, self).__init__()
        self.g = nn.RNN(input_size=INPUT_DIM, hidden_size=HIDDEN_SIZE, num_layers=HIDDEN_LAYERS, nonlinearity='tanh')
        self.memory = SlotMemory(beta=BETA)
        last_hidden = torch.zeros((HIDDEN_LAYERS, HIDDEN_SIZE)).to(torch.float64)
        self.register_buffer("last_hidden", last_hidden)

    def forward(self, inp: torch.Tensor, 
                x_obs: torch.Tensor = None, 
                train: bool = False, 
                device: torch.DeviceObjType = None,
                seq_len: int = SEQ_LEN) -> List[torch.Tensor]:
        # g_cur = self.activation(g_cur)
        if train:
            # if self.memory.empty() is False:
            #     idx = random.randint(1, seq_len - 2)
            #     g_cur_init, h = self.g(inp[:idx, :], self.last_hidden)
            #     self.last_hidden = self.memory.retrieve_g(x_obs[idx], OUTPUT_DIM).unsqueeze(0)
            #     # print(f'hidden: {self.last_hidden}')
            #     g_cur_cued, h = self.g(inp[idx:, :], self.last_hidden)
            #     g_cur = torch.vstack([g_cur_init, g_cur_cued])
            #     # print(g_cur.shape)
            #     # input(':')
            # else:
            g_cur, h = self.g(inp, self.last_hidden)
            # print(g_cur)
            if self.memory.empty() is False:
                prob_g = self.memory.retrieve_prob(g_cur, output_dim=OUTPUT_DIM)
                prob_x = self.memory.retrieve_prob(x_obs, output_dim=OUTPUT_DIM, index_g=False)
            else:
                prob_g = prob_x = torch.full_like(g_cur, 1.0 / g_cur.size(0), requires_grad=True)
            p_new = torch.hstack([x_obs, g_cur])
            self.memory.store(p_new)
            return prob_g, prob_x, g_cur, h 

        else:
            g_cur, h = self.g(inp, self.last_hidden)
            x_p = self.memory.retrieve_x(g_cur, output_dim=OUTPUT_DIM)
            return x_p

def compute_loss(prob_g: torch.Tensor,
                 prob_x: torch.Tensor
                 ) -> torch.Tensor:
    
    ce_loss = nn.CrossEntropyLoss(reduction='sum')
    return ce_loss(prob_g, prob_x)

def compute_sparsity_loss(g):
    return LAMBDA * nn.MSELoss()(g, torch.zeros_like(g))

def train(writer: SummaryWriter):
    device = torch.device('cuda:1')
    # Initialize model
    model = GridModule().to(torch.float64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=GAMMA)

    torch.autograd.set_detect_anomaly(True)
    global_iter = 0 
    for iter in range(1, 10000):
        total_loss = 0
        avg_val_acc = 0.0
        for env in range(1, NUM_ENVIRONMENTS + 1):
            # Generate Graph
            g = Graph(GRID_ROWS, GRID_COLS)
            # g.printGrid()
            
            # print('Creating training samples')
            X_train, y_train, hidden_paths = g.randomWalk(NUM_SAMPLES, NUM_HIDDEN, NUM_BATCHES)
            
            # Generate Test samples
            # print('Generating Test samples')
            X_test, y_test = g.generateTest(hidden_paths, pad=True)

            # Typecast to tensors
            X_train, y_train = torch.tensor(X_train, requires_grad=False).to(device), torch.tensor(y_train, requires_grad=False).to(device)
            X_train, y_train = X_train.squeeze(), y_train.squeeze()

            X_test = torch.from_numpy(X_test).to(torch.float64).to(device)
            y_test = torch.from_numpy(y_test).to(torch.float64).to(device)
            env_iter = 0
            # print('Starting training')

            for i in range(0, len(X_train) - SEQ_LEN + 1, SEQ_LEN):   
                inp = X_train[i:i + SEQ_LEN, :]    
                target = y_train[i:i + SEQ_LEN, :]    

                prob_g, prob_x, g, h = model(inp, target, train=True, device=device)
                loss = compute_loss(prob_g, prob_x)
                if iter > 1000:
                    loss += compute_sparsity_loss(g)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                total_loss += loss.iGridModule()

                _, h = model.g(inp)
                model.last_hidden = h.detach()
                env_iter += 1
                global_iter += 1

            # perform validation testing over hidden paths
            if iter > 1500:
                for test_seq in range(len(X_test)):
                    x_p = model(X_test[test_seq], train=False, device=device, seq_len=len(X_test[test_seq]))
                    out_max_idx = torch.argmax(x_p[-1])
                    y_max_idx = torch.argmax(y_test[test_seq][-1])
                    if out_max_idx == y_max_idx:
                        avg_val_acc += 1.0
            
            # reset hidden state
            model.last_hidden = torch.zeros((HIDDEN_LAYERS, HIDDEN_SIZE)).to(torch.float64).to(device)

            # model.memory.forget(FORGET_PCT)
        
            # Once training is done in this environment, reset memories
            model.memory.patterns = None
        
        lr = scheduler.get_last_lr()
        if iter % 5 == 0:
            print(f'iter {iter:3d} | lr {lr} | loss {total_loss:5.2f}')
        writer.add_scalar(f'Loss', total_loss, iter)
        total_loss = 0.0
    # Save generalizing entorhinal representations
    torch.save(model.state_dict(), 'checkpoints/model_6x6_grid_h10_tanh_sparse_5')

if __name__ == '__main__':
    writer = SummaryWriter()
    writer.add_hparams({'lr': LEARNING_RATE, 
                        'Grid Rows': GRID_ROWS,
                        'Grid Cols': GRID_COLS,
                        'HIDDEN_SIZE': HIDDEN_SIZE,
                        'SEQ_LEN': SEQ_LEN,
                        'NUM_ENVIRONMENTS': NUM_ENVIRONMENTS,
                        'LAMBDA': LAMBDA,
                        'BETA': BETA,
                        'NUM_EPOCHS': NUM_EPOCHS
                        }, {})
    train(writer)