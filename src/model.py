import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

LEARNING_RATE = 0.001
EPOCHS = 2

class ArielModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ArielModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.output_mu = nn.Linear(256, output_dim)  
        self.output_sigma = nn.Linear(256, output_dim)  

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))

        mu = self.output_mu(x)  
        sigma = torch.exp(self.output_sigma(x)) 
        
        return mu, sigma



def gaussian_log_likelihood(y, mu, sigma):
    gll = -0.5 * (torch.log(2 * torch.tensor(np.pi).to(y.device)) + 
                   torch.log(sigma**2) + 
                   ((y - mu)**2) / (sigma**2))
    return gll.sum(dim=1)


def train_model(model, train_loader, val_loader, num_epochs=EPOCHS, learning_rate=LEARNING_RATE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc="training"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            mu, sigma = model(inputs)
            
            gll = gaussian_log_likelihood(targets, mu, sigma)
            loss = -gll.mean()

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in tqdm(val_loader, desc="validation"):
                    inputs, targets = inputs.to(device), targets.to(device)
                    mu, sigma = model(inputs)
                    gll = gaussian_log_likelihood(targets, mu, sigma)
                    val_loss += -gll.mean().item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), 'ariel_model.pth')
    print("model saved.")
