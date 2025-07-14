import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score

print("Starting semi-supervised learning for Leishmania compound activity prediction...")

# 1. Load the dataset of compounds active and inactive against Leishmania
active_df = pd.read_csv("active_leishmania_compounds_CHEMBL_for_training.csv")
nonactive_df = pd.read_csv("active_NOT_leishmania_compounds_CHEMBL_for_training.csv")

# Extract unique SMILES for active and inactive compounds
active_smiles = list(set(active_df['canonical_smiles'].dropna()))
nonactive_smiles = list(set(nonactive_df['canonical_smiles'].dropna()))

# Ensure both lists are equal length (for balanced classes)
if len(nonactive_smiles) > len(active_smiles):
    nonactive_smiles = nonactive_smiles[:len(active_smiles)]
if len(active_smiles) > len(nonactive_smiles):
    active_smiles = active_smiles[:len(nonactive_smiles)]

print(f"Active compounds (unique): {len(active_smiles)}")
print(f"Inactive compounds (unique): {len(nonactive_smiles)}")

# 2. Define a function to compute Morgan fingerprint (radius 2, 2048-bit) using RDKit
def get_fingerprint(smiles: str):
    """Convert a SMILES string to a Morgan fingerprint vector (np.array of 2048 bits)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Invalid SMILES
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros((2048,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)  # convert fingerprint to numpy array of 0/1
    return arr

# Compute fingerprints for all molecules, building the dataset
X_data = []
y_labels = []
# Positive class (active)
for smi in active_smiles:
    fp = get_fingerprint(smi)
    if fp is not None:
        X_data.append(fp)
        y_labels.append(1)
# Negative class (inactive)
for smi in nonactive_smiles:
    fp = get_fingerprint(smi)
    if fp is not None:
        X_data.append(fp)
        y_labels.append(0)

X_data = np.array(X_data)
y_labels = np.array(y_labels)
print(f"Total compounds with fingerprints: {X_data.shape[0]}")

# 3. Split into training and test sets for evaluation of the classifier later
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_data, y_labels, stratify=y_labels, test_size=0.2, random_state=42
)
print(f"Train set size: {X_train_full.shape[0]}, Test set size: {X_test_full.shape[0]}")

# For unsupervised training (autoencoder), we will **only use the training set** fingerprints
X_train = X_train_full  # unlabeled data for autoencoder training
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

# 4. Define the Autoencoder model architecture
class Autoencoder(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, code_dim=128):
        super(Autoencoder, self).__init__()
        # Encoder: compress input to latent code
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, code_dim),
            nn.ReLU()
        )
        # Decoder: reconstruct input from latent code
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Sigmoid outputs in [0,1] for each bit
        )
    def forward(self, x):
        # Pass data through encoder and then decoder
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed
    def encode(self, x):
        # Utility to get the latent code from input
        return self.encoder(x)

# Initialize autoencoder
ae = Autoencoder(input_dim=2048, hidden_dim=512, code_dim=256)
print(f"Autoencoder initialized: encoding 2048 -> 256 dimensions")

# Define loss and optimizer for the autoencoder
criterion_ae = nn.BCELoss()  # binary cross-entropy loss, good for reconstructing 0/1 bits
optimizer_ae = torch.optim.Adam(ae.parameters(), lr=0.001)

# 5. Train the autoencoder (unsupervised) on X_train
num_epochs = 1000
for epoch in range(1, num_epochs+1):
    # Forward pass and loss
    recon = ae(X_train_tensor)
    loss = criterion_ae(recon, X_train_tensor)
    # Backpropagation
    optimizer_ae.zero_grad()
    loss.backward()
    optimizer_ae.step()
    # Print progress occasionally
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d}/{num_epochs}, Reconstruction Loss: {loss.item():.4f}")

# Save the trained autoencoder model for future use
torch.save(ae.state_dict(), "autoencoder_model.pth")

# 6. Generate latent feature vectors for train and test sets using the trained encoder
ae.eval()  # set autoencoder to evaluation mode
with torch.no_grad():
    X_train_latent = ae.encode(torch.tensor(X_train_full, dtype=torch.float32)).numpy()
    X_test_latent  = ae.encode(torch.tensor(X_test_full, dtype=torch.float32)).numpy()

print(f"Latent feature shape: {X_train_latent.shape} (should be [n_samples, 128])")

class ActivityClassifier(nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)    # logit
        )
    def forward(self, x):
        return self.net(x)

clf = ActivityClassifier(input_dim=X_train_latent.shape[1])
criterion_clf = nn.BCEWithLogitsLoss()  # binary cross-entropy loss with logits
# L2 weight-decay: set weight_decay > 0
optimizer_clf = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)

# Convert latent features and labels to tensors for training
X_train_latent_tensor = torch.tensor(X_train_latent, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

# 8. Train the classifier on the training set latent features
num_epochs_clf = 1000
for epoch in range(1, num_epochs_clf+1):
    clf.train()
    # Forward pass
    logits = clf(X_train_latent_tensor)
    loss_clf = criterion_clf(logits, y_train_tensor)
    # Backpropagation
    optimizer_clf.zero_grad()
    loss_clf.backward()
    optimizer_clf.step()
    # Print training progress occasionally
    if epoch % 5 == 0 or epoch == 1:
        # Compute training accuracy for info
        preds = (torch.sigmoid(logits) >= 0.5).int()
        train_acc = (preds.numpy() == y_train.reshape(-1,1)).mean()
        print(f"Epoch {epoch:02d}/{num_epochs_clf}, Loss: {loss_clf.item():.4f}, Train Accuracy: {train_acc:.3f}")

# 9. Evaluate the classifier on the test set
clf.eval()
with torch.no_grad():
    X_test_latent_tensor = torch.tensor(X_test_latent, dtype=torch.float32)
    test_logits = clf(X_test_latent_tensor)
    # Apply sigmoid to get probabilities, then threshold at 0.5 for class prediction
    test_probs = torch.sigmoid(test_logits).numpy().flatten()
    y_pred = (test_probs >= 0.5).astype(int)

# Print evaluation metrics
print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("Classification Report (Test Set):")
print(classification_report(y_test, y_pred, digits=4))
