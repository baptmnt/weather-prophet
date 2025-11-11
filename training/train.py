# Le but de ce fichier est d'offrir une function permettant d'entrainer
# un modèle, en faisant varier des paramètres (epoch, batch size, learning rate, etc.)
# Tout en exportant les résultats dans TensorBoard

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def mae(pred, target):
    return torch.mean(torch.abs(pred - target))

def r2_score(pred, target):
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target, dim=0)) ** 2)
    return 1 - ss_res / ss_tot


def train_model(name, model:nn.Module, dataset, train_size=0.8, val_size=0.1, num_epochs=10, batch_size=32, learning_rate=1e-3):
    # Division du dataset en train, val, test
    train_size = int(train_size * len(dataset))
    val_size = int(val_size * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # Création des DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #date = datetime.now().strftime("%Y%m%d-%H%M%S")

    writer = SummaryWriter(log_dir=f"runs/{name}_lr{learning_rate}_bs{batch_size}_ep{num_epochs}")

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_mae, train_r2 = 0.0, 0.0, 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_mae  += mae(outputs, targets).item() * inputs.size(0)
            train_r2   += r2_score(outputs, targets).item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_mae  /= len(train_loader.dataset)
        train_r2   /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss, val_mae, val_r2 = 0.0, 0.0, 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                val_mae  += mae(outputs, targets).item() * inputs.size(0)
                val_r2   += r2_score(outputs, targets).item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_mae  /= len(val_loader.dataset)
        val_r2   /= len(val_loader.dataset)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metric/train_MAE", train_mae, epoch)
        writer.add_scalar("Metric/val_MAE", val_mae, epoch)
        writer.add_scalar("Metric/train_R2", train_r2, epoch)
        writer.add_scalar("Metric/val_R2", val_r2, epoch)


        print(f"Époque [{epoch+1}/{num_epochs}]  |  Perte train: {train_loss:.4f}  |  Perte val: {val_loss:.4f}")
    
    writer.close()

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
    test_loss /= len(test_loader.dataset)
    print(f"Perte test : {test_loss:.4f}")