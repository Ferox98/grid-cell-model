import torch
import torch.nn as nn 
import torch.optim as optim 

from src.models.graph import Graph 
from src.conf import *
from src.utils import get_batch

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=INPUT_DIM, hidden_size=HIDDEN_SIZE, num_layers=HIDDEN_LAYERS)
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_DIM)
        # self.softmax = nn.Softmax(dim=2)

        last_hidden = torch.zeros((HIDDEN_LAYERS, NUM_BATCHES, HIDDEN_SIZE)).to(torch.float64)
        cell_state = torch.zeros_like(last_hidden)
        
        self.register_buffer("last_hidden", last_hidden)
        self.register_buffer("cell_state", cell_state)

    def forward(self, x):
        out, h = self.lstm(x, (self.last_hidden, self.cell_state))
        fc = self.fc(out)
        # softmax = self.softmax(fc)
        return fc, h

def train():
    # Generate Graph
    print("Generating graph")
    g = Graph(GRID_ROWS, GRID_COLS)
    g.printGrid()
    
    print('Creating training samples')
    X_train, y_train, hidden_paths = g.randomWalk(NUM_SAMPLES, NUM_HIDDEN, NUM_BATCHES)
    
    # Generate Test samples
    print('Generating Test samples')
    X_test, y_test = g.generateTest(hidden_paths, pad=True)

    # Typecast to tensors
    X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
    # with np.printoptions(threshold=np.inf):
    #     print(X_train[0, 0:10, :].numpy())
    #     print(y_train[0, 0:10, :].numpy())
    #     input(':')

    X_test, y_test = torch.from_numpy(X_test).to(torch.float64), torch.from_numpy(y_test).to(torch.float64)
    
    device = torch.device('cuda:2')

    # Transform into (sequence, batch, feature) format
    X_train = torch.transpose(X_train, 0, 1).contiguous().to(device) 
    y_train = torch.transpose(y_train, 0, 1).contiguous().to(torch.float64).to(device)
    X_test = torch.transpose(X_test, 0, 1).to(device)
    y_test = torch.transpose(y_test, 0, 1).to(device)

    # Initialize model
    model = LSTM().to(torch.float64).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.9)
   
    print('Starting training')
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch, i in enumerate(range(0, X_train.size(0) - 1, SEQ_LEN)):
            data, targets = get_batch(X_train, i), get_batch(y_train, i)
            output, (h, c) = model(data) 
            # print(targets.shape)
            if epoch > 0 and epoch % 50 == 0 and i == 0:        
                print(torch.argmax(output[:,0,:], dim=1))
                print(torch.argmax(targets[:,0,:], dim=1))
                print('')
            loss = loss_fn(output, targets) #+ ALPHA * torch.linalg.vector_norm(h, ord=1)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss  

            output, (h, c) = model(data) 
            model.cell_state = c.detach()
            model.last_hidden = h.detach()      

        if epoch % 10 == 0:
            # reset hidden states with appropriate dimensions
            test_hidden = torch.zeros((HIDDEN_LAYERS, NUM_HIDDEN, HIDDEN_SIZE)).to(torch.float64).to(device)
            test_cell_state = torch.zeros_like(test_hidden).to(device)
            model.last_hidden = test_hidden 
            model.cell_state = test_cell_state 

            out, _ = model(X_test.to(device))
            # check if model's last output is same as last (hidden) observation
            out_max_idx = torch.argmax(out[-1,:,:], dim=1)
            y_max_idx = torch.argmax(y_test[-1,:,:], dim=1)
            accurate = (out_max_idx == y_max_idx).nonzero().size(0)
            lr = scheduler.get_last_lr()
            print(f'epoch {epoch:3d} | lr {lr} | loss {total_loss:5.2f} | validation accuracy: {accurate / len(X_test[0])}')
        
        # reset hidden state
        model.last_hidden = torch.zeros((HIDDEN_LAYERS, NUM_BATCHES, HIDDEN_SIZE)).to(torch.float64).to(device)
        model.cell_state = torch.zeros_like(model.last_hidden).to(device)
        scheduler.step()

if __name__ == '__main__':
    train()