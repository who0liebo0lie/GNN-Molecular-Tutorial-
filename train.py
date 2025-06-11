import torch
from model import GCN
from data_loader import load_data

def train():
    train_loader, test_loader = load_data()
    model = GCN(num_features=11, hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    for epoch in range(10):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y[:, 0:1])
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Save model for deployment
    torch.save(model.state_dict(), "gnn_model.pt")

if __name__ == "__main__":
    train()
