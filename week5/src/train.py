import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from tqdm import tqdm


DEFAULT_EPOCHS = 10
DEFAULT_LR = 1e-3


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(loader, leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    return acc, f1, cm


def train_loop(model, dataloaders_factory, device, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR):
    fold_accuracies = []
    fold_f1s = []
    
    for fold in range(1, 11):
        print(f"=== Fold {fold} ===")

        train_loader, test_loader = dataloaders_factory.get_dataloaders(fold)

        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")

        acc, f1, cm = evaluate(model, test_loader, device)
        # TODO Report cm
        print(f"Fold {fold} | Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
        fold_accuracies.append(acc)
        fold_f1s.append(f1)

    print("=== 10-fold CV Results ===")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"Mean Macro F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

    # TODO Return thee metrics from the function
