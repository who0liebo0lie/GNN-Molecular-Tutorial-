import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)  # Regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        #print statement:
        print(f"Shape of x BEFORE global_mean_pool: {x.shape}")
        x = global_mean_pool(x, batch)
        # print statement:
        print(f"Shape of x AFTER global_mean_pool: {x.shape}")
        
        #return self.lin(x)x = global_mean_pool(x, batch)
        return self.lin(x)
