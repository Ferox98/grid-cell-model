import math
import time

import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from graph import Graph

import numpy as np

from ..conf import *
from ..utils import batchify, get_batch

torch.set_printoptions(linewidth=2000)

g = Graph(GRID_ROWS, GRID_COLS)
g.printGrid()
print('creating training samples')
X_train, y_train, hidden_paths = g.randomWalk(NUM_SAMPLES, NUM_HIDDEN)
print('creating test samples')
test_samples = g.generateTest(hidden_paths)
X_test, y_test = test_samples[:, 0], test_samples[:, 1]


class TransformerModel(torch.nn.Module):

    def __init__(self, n_landmarks: int, n_dirs: int, d_model: int, n_head: int, d_hid: int,
                 n_layers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layers = TransformerEncoderLayer(d_model, n_head, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.d_model = d_model 
        self.linear_in = torch.nn.Linear(n_dirs, d_model)
        self.linear_out = torch.nn.Linear(d_model, n_landmarks)
        self.softmax = torch.nn.Softmax()

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear_in.bias.data.zero_()
        self.linear_in.weight.data.uniform_(-initrange, initrange)
        self.linear_out.bias.data.zero_()
        self.linear_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor=None) -> torch.tensor:
        # print(src.shape)
        src = self.linear_in(src)
        # print(src.shape)
        src = self.pos_encoder(src)
        # print(src.shape)
        output = self.transformer_encoder(src, src_mask)
        # print(output.shape)
        output = self.linear_out(output) 
        # print(output.shape)
        # input(':')
        # output = self.softmax(output)
        return output 

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, s_len: int=5000, dropout: float=0.1):
        super().__init__() 
        self.dropout = torch.nn.Dropout(p=dropout)
        position = torch.arange(s_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, s_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
X_test, y_test = torch.tensor(X_test.tolist()).to(torch.float64), torch.tensor(y_test.tolist()).to(torch.float64)

X_train, y_train = batchify(X_train, INPUT_DIM).to(torch.float64), batchify(y_train, OUTPUT_DIM).to(torch.float64)

# Initialize model params
d_model = 16
d_hid = 16
n_layers = 2
n_head = 2
dropout = 0

# Initialize model
model = TransformerModel(OUTPUT_DIM, INPUT_DIM, d_model, n_head, d_hid, n_layers, dropout).to(torch.float64)
criterion = torch.nn.CrossEntropyLoss()
lr = 1.0
optimizer = torch.optim.Adam(model.parameters())
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma = 0.97)

def train(model: torch.nn.Module) -> None:
    model.train()
    total_loss = 0.0
    log_interval = 16
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        for batch, i in enumerate(range(0, NUM_BATCHES, BATCHES_PER_EPOCH)):
            data, targets = get_batch(X_train, i), get_batch(y_train, i)
            output = model(data)

            loss = criterion(output, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            if i ==0 and epoch % 50 == 0:
                # lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                print(f'epoch {epoch:3d} | {i:5d}/{NUM_BATCHES:5d} batches | '
                        f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                        f'loss {cur_loss:5.2f}')
                total_loss = 0
                start_time = time.time()
        
        # scheduler.step()


if __name__ == '__main__':
    train(model)