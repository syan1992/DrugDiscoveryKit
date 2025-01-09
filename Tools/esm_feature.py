import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import esm
import pandas as pd
import pdb
import numpy as np

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model = model
model.eval()  # disables dropout for deterministic results

# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
train_csv = pd.read_csv('train-protein.csv')
test_csv = pd.read_csv('test-protein.csv')

train_csv = train_csv[['Uniprot', 'Sequence']]
test_csv = test_csv[['Uniprot', 'Sequence']]

csv = pd.concat([train_csv, test_csv], ignore_index=True)
csv = csv.drop_duplicates()

data = list(csv.itertuples(index=False, name=None)) #train_data + test_data
'''
data = [
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein3",  "K A <mask> I S Q"),
]
'''
batch_labels, batch_strs, batch_tokens = batch_converter(data[0:10])
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=False)
token_representations = results["representations"][33]

# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
sequence_representations = []
protein_features = {}
for i, tokens_len in enumerate(batch_lens):
    repres = token_representations[i, 1 : tokens_len - 1].mean(0)
    sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    protein_features[batch_labels[i]] = repres.detach().cpu().numpy()

np.save('protein_features.npy', protein_features)
