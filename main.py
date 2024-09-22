from src.model import ArielModel, train_model
from src.dataloader import split_data
import torch


if __name__ == '__main__':

    train_dataloader, val_dataloader = split_data(
        airs_path=r"C:\Users\SKYMIL\Desktop\vscode\kaggle\ariel_exoplanets\data\airs_v4.npy",
        fgs_path=r"C:\Users\SKYMIL\Desktop\vscode\kaggle\ariel_exoplanets\data\fgs_v4.npy",
        labels_path=r"C:\Users\SKYMIL\Desktop\vscode\kaggle\ariel_exoplanets\data\train_labels.csv"
    )
    # batch = Shape of x: torch.Size([32, 600]), Shape of y: torch.Size([32, 283])
    input_dims = 600
    output_dims = 283
    model = ArielModel(input_dims, output_dims)

    train_model(model, train_dataloader, val_dataloader)
    