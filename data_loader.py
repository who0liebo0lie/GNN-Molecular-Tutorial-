from torch_geometric.datasets import QM9
from torch_geometric.transforms import AddSelfLoops
from torch_geometric.data import DataLoader

def load_data(batch_size=32):
    dataset = QM9(root='data/QM9', transform=AddSelfLoops())
    dataset = dataset.shuffle()
    train_dataset = dataset[:10000]
    test_dataset = dataset[10000:11000]
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

def get_sample_graph():
    import torch
    from torch_geometric.data import Data

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    x = torch.tensor([[1], [2]], dtype=torch.float)

    return Data(x=x, edge_index=edge_index)
