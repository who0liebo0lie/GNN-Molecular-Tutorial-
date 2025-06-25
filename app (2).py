import io
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import torch
from model import GCN
from data_loader import get_sample_graph
import pandas as pd
from torch_geometric.data import Batch

st.set_page_config(page_title="💗 Molecular GNN", page_icon="🧬")
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        background-color: #fff0f5;
        color: #800080;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    .stButton>button {
        background-color: #ff69b4;
        color: white;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💖 GNN Molecular Property Predictor")
st.write("Upload a SMILES string and visualize your molecule below. The GNN will predict a molecular property.")

smiles = st.text_input("🧪 Enter SMILES string:", "CCO")

mol = Chem.MolFromSmiles(smiles)
if mol:
    if mol.GetNumAtoms() < 2:
        st.warning("Molecule has too few atoms to form a valid graph. Try a more complex structure.")
    else:
        st.image(Draw.MolToImage(mol, size=(300, 300)), caption="Molecule Preview")
        sample = get_sample_graph(smiles)

        # Make sure sample is not None and has the right feature shape
        if sample is not None and sample.x.shape[1] == 11:
            try:
                sample_batch = Batch.from_data_list([sample])
                model = GCN(num_features=11, hidden_channels=64)
                state_dict = torch.load("gnn_model.pt", map_location=torch.device("cpu"))
                model.load_state_dict(state_dict)
                model.eval()

                pred_value = model(sample_batch).item()
                st.success(f"Predicted Property: {pred_value:.4f}")

                df = pd.DataFrame([[smiles, pred_value]], columns=["SMILES", "Predicted_Property"])
                csv = df.to_csv(index=False)
                st.download_button("📥 Download Prediction as CSV", data=csv, file_name="prediction.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Model prediction failed: {e}")
        else:
            st.warning("Unable to process this molecule. It may have invalid or mismatched features.")
else:
    st.warning("Invalid SMILES string.")
st.write(f"Node feature shape: {sample.x.shape}")
