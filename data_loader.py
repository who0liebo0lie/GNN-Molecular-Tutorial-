from torch_geometric.datasets import QM9
from torch_geometric.transforms import AddSelfLoops
from torch_geometric.data import DataLoader

import torch
from torch_geometric.data import Data

from rdkit import Chem
from torch_geometric.data import Data



def atom_features(atom):
    try:
        return torch.tensor([
            float(atom.GetAtomicNum()),              # 1
            float(atom.GetDegree()),                 # 2
            float(atom.GetFormalCharge()),           # 3
            float(int(atom.GetHybridization())),     # 4
            float(atom.GetTotalNumHs()),             # 5
            float(atom.GetImplicitValence()),        # 6
            float(int(atom.GetIsAromatic())),        # 7
            float(atom.GetMass()),                   # 8
            float(atom.GetNumRadicalElectrons()),    # 9
            float(int(atom.GetChiralTag())),         # 10
            float(atom.GetTotalDegree())             # 11
        ], dtype=torch.float32)
    except Exception as e:
        print(f"[atom_features] Error creating features: {e}")
        return torch.zeros(11, dtype=torch.float32)


def get_sample_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Extract atom features
    node_feats = []
    for i, atom in enumerate(mol.GetAtoms()):
        feat = atom_features(atom)
        if feat.shape[0] != 11:
            print(f"[DEBUG] Atom {i} has {feat.shape[0]} features: {feat}")
            return None  # skip if features are malformed
        node_feats.append(feat)

    if len(node_feats) == 0:
        print("[DEBUG] No atom features found")
        return None

    x = torch.stack(node_feats)

    # Build edge index
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

def load_data(batch_size=32):
    dataset = QM9(root='data/QM9', transform=AddSelfLoops())
    dataset = dataset.shuffle()
    train_dataset = dataset[:10000]
    test_dataset = dataset[10000:11000]
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader


