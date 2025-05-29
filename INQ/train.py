# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Assuming inq_sgd.py and inq_scheduler.py are in the same directory
from inq_sgd import INQSGD
from inq_scheduler import INQScheduler, reset_lr_scheduler


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Loading and Preprocessing for MNIST
    # ResNet-18 expects 3-channel images. MNIST is 1-channel.
    # We will convert MNIST to 3 channels by repeating the single channel.
    # Also, resize to 224x224 for ResNet-18 input.
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Convert 1-channel to 3-channel
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST normalization
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 2. Model Initialization (ResNet-18)
    # Load pre-trained ResNet-18 and modify the final layer for MNIST (10 classes)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # Use latest weights argument
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10) # 10 classes for MNIST
    model = model.to(device)

    # Freeze all layers except the final classification layer initially if needed
    # for transfer learning, but for INQ we usually want to quantize all layers.
    # So, we keep them trainable.

    # 3. INQ Configuration
    # Example iterative steps (portions of weights to be quantized)
    iterative_steps = [0.5, 0.75, 1.0] # Quantize 50%, then 25% more, then final 25%
    inq_strategy = "pruning" # or "random"
    weight_bits = 4 # Example: 4-bit quantization

    # Set initial_lr in optimizer's defaults for scheduler reset
    initial_lr = 0.01
    optimizer = INQSGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=1e-4, weight_bits=weight_bits)
    
    # Store initial_lr in param_groups for easy reset by scheduler
    for group in optimizer.param_groups:
        group['initial_lr'] = group['lr']


    inq_scheduler = INQScheduler(optimizer, iterative_steps, strategy=inq_strategy)

    # Learning Rate Scheduler for fine-tuning within each INQ step
    # This scheduler will be reset at each INQ step
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    criterion = nn.CrossEntropyLoss()

    # Initial evaluation before INQ
    print("\n--- Initial Evaluation (Full Precision) ---")
    loss, acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Initial Test Loss: {loss:.4f}, Initial Test Acc: {acc:.4f}")

    # INQ Training Loop
    fine_tune_epochs_per_inq_step = 5 # Number of epochs to fine-tune after each INQ step

    for inq_step_idx in range(len(iterative_steps)):
        print(f"\n--- INQ Step {inq_step_idx + 1}/{len(iterative_steps)} (Quantizing {iterative_steps[inq_step_idx]*100:.0f}% of weights) ---")

        # 1. Reset LR scheduler
        reset_lr_scheduler(lr_scheduler)

        # 2. Perform weight partitioning and quantization for the current INQ step
        inq_scheduler.step() # This calls `quantize()` internally

        # Evaluate immediately after quantization (before fine-tuning)
        print(f"--- Evaluation after INQ Step {inq_step_idx + 1} Quantization (before fine-tuning) ---")
        loss_after_quant, acc_after_quant = evaluate_model(model, test_loader, criterion, device)
        print(f"Test Loss: {loss_after_quant:.4f}, Test Acc: {acc_after_quant:.4f}")

        # 3. Fine-tune the model
        print(f"--- Fine-tuning for {fine_tune_epochs_per_inq_step} epochs ---")
        for epoch in range(fine_tune_epochs_per_inq_step):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
            lr_scheduler.step() # Step the LR scheduler within the fine-tuning loop
            print(f"Epoch {epoch+1}/{fine_tune_epochs_per_inq_step} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        print(f"--- Evaluation after INQ Step {inq_step_idx + 1} Fine-tuning ---")
        final_loss, final_acc = evaluate_model(model, test_loader, criterion, device)
        print(f"Final Test Loss: {final_loss:.4f}, Final Test Acc: {final_acc:.4f}")


    print("\n--- INQ Procedure Completed ---")
    final_test_loss, final_test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Overall Final Test Loss: {final_test_loss:.4f}, Overall Final Test Acc: {final_test_acc:.4f}")


if __name__ == "__main__":
    main()