import torch 
from tem.src.models.model import GridModule 
from src.models.graph import Graph 
from src.conf import * 
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
device = torch.device('cuda:0')
# load 4x4 model
model = GridModule().to(torch.float64).to(device)
model.load_state_dict(torch.load('checkpoints/model_6x6_grid_h10_tanh_sparse_5'))


g = Graph(GRID_ROWS, GRID_COLS, shuffle=False)
# Peform a random walk of length 1000
X_sample, y_sample, _ = g.randomWalk(5000, 0, NUM_BATCHES)
X_sample, y_sample = torch.tensor(X_sample).to(device).squeeze(), torch.tensor(y_sample).to(device).squeeze()
y_labels = y_sample.argmax(dim=1)
out, _ = model.g(X_sample)
print(out[:10, :])
out = torch.abs(out)
# out = torch.clamp(out, min=0)
firing_rates = {}
# for each grid code, get average firing rate over a location
for i in range(len(y_labels)):
    cur_rate = out[i]
    cur_label = y_labels[i]
    for j in range(len(cur_rate)):
        if j not in firing_rates.keys():
            firing_rates[j] = [0.0] * 36
        firing_rates[j][cur_label] += out[i][j].item()

colors = [(0, 'blue'), (0.5, 'orange'), (1, 'red')]
custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)
for i in range(10):
    fig, ax = plt.subplots()
    mat = torch.tensor(firing_rates[i]).reshape(6, 6)
    mat = torch.nn.functional.normalize(mat)
    # mat = mat / 5000.0
    im = ax.imshow(mat, cmap='viridis', interpolation='bilinear')
    cbar = plt.colorbar(im)
    plt.savefig(f'neuron_{i}.png')