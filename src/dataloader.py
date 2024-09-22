import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

BATCH_SIZE = 32
TEST_SIZE = 0.2
NUM_WORKERS = 8
AIRS_COMP = 200
FGS_COMP = 400

class ArielDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx])
        if self.y is not None:
            y = torch.tensor(self.y[idx])
        else:
            y = torch.tensor([0] * 283)
        return x, y
    
def process_data(data1, data2):
    scaler_airs = StandardScaler()
    scaler_fgs = StandardScaler()
    data1 = scaler_airs.fit_transform(data1)
    data2 = scaler_fgs.fit_transform(data2)

    pca_airs = PCA(n_components=AIRS_COMP)  
    pca_fgs = PCA(n_components=FGS_COMP)  
    data1 = pca_airs.fit_transform(data1)
    data2 = pca_fgs.fit_transform(data2)

    X_train = np.hstack((data1, data2))

    return X_train

def split_data(airs_path, fgs_path, labels_path):
    data_train_airs = np.load(airs_path)
    data_train_fgs = np.load(fgs_path)
    train_labels = np.loadtxt(labels_path, delimiter = ',', skiprows = 1)
    targets = train_labels[:,1:]
    X_train = process_data(data_train_airs, data_train_fgs)
    X_train, X_val, y_train, y_val = train_test_split(X_train, targets, test_size=TEST_SIZE, random_state=42)
    train_dataloader = torch.utils.data.DataLoader(ArielDataset(X_train, y_train), num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)
    valid_dataloader = torch.utils.data.DataLoader(ArielDataset(X_val, y_val), num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)

    return train_dataloader, valid_dataloader