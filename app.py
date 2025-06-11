import streamlit as st
import torch
from model import GCN
from torch_geometric.datasets import QM9
from torch_geometric.transforms import AddSelfLoops
from torch_geometric.data import DataLoader


st.title("GNN Molecular Property Prediction")

model = GCN(num_features=11, hidden_channels=64)
model.load_state_dict(torch.load("gnn_model.pt", map_location=torch.device('cpu')))
model.eval()

st.markdown("Predict molecular properties from the QM9 dataset using a pretrained GNN.")

# Load dataset and sample molecule
dataset = QM9(root='data/QM9', transform=AddSelfLoops())
sample = dataset[0]
loader = DataLoader([sample], batch_size=1)
sample_batch = next(iter(loader))

with torch.no_grad():
    pred = model(sample_batch)


st.write(f"**Predicted property value:** {pred.item():.4f}")
