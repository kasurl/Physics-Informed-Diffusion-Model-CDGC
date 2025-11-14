import torch
import torch.optim as optim
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
from model import UNet
from data_loader import create_data_loaders


def calculate_iou(pred, target):
    """Calculate Intersection over Union"""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return intersection / (union + 1e-8)


def save_model_if_best(model, epoch, val_loss, save_dir="saved_models"):
    """Save model if it has the best validation loss"""
    os.makedirs(save_dir, exist_ok=True)

    # Simple implementation - save every model
    model_path = os.path.join(save_dir, f'epoch_{epoch}_val_loss_{val_loss:.4f}.pth')
    torch.save(model.state_dict(), model_path)
    return model_path


def train_model(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
                model_save_dir="saved_models", batch_size=4, num_epochs=50, learning_rate=0.001):
    """Main training function"""

    # Create data loaders
    train_dataloader, val_dataloader = create_data_loaders(
        train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, batch_size
    )

    # Model setup
    model = UNet().cuda()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training metrics
    train_losses, val_losses, val_accuracies, val_ious = [], [], [], []

    print(f"Starting training for {num_epochs} epochs")
    print(f"Training images: {len(train_dataloader.dataset)}")
    print(f"Validation images: {len(val_dataloader.dataset)}")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)

        train_epoch_loss = running_train_loss / len(train_dataloader.dataset)
        train_losses.append(train_epoch_loss)

        # Validation phase
        model.eval()
        running_val_loss, total_iou, correct, total = 0.0, 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                labels_float = labels.float()

                val_loss = criterion(outputs, labels_float)
                running_val_loss += val_loss.item() * inputs.size(0)

                predicted = (outputs > 0.4).float()
                total += labels_float.numel()
                correct += (predicted == labels_float).sum().item()

                iou = calculate_iou(predicted, labels_float)
                total_iou += iou.item()

        val_epoch_loss = running_val_loss / len(val_dataloader.dataset)
        val_losses.append(val_epoch_loss)

        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)

        val_iou = total_iou / len(val_dataloader)
        val_ious.append(val_iou)

        # Save model
        model_path = save_model_if_best(model, epoch, val_epoch_loss, model_save_dir)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(
            f'Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Accuracy: {val_accuracy:.2f}% | Val IoU: {val_iou:.4f}')
        print(f'Model saved: {model_path}')

    # Plot results
    plot_training_curves(train_losses, val_losses, val_accuracies, val_ious)

    # Save metrics
    save_training_metrics(train_losses, val_losses, val_accuracies, val_ious)

    return model


def plot_training_curves(train_losses, val_losses, val_accuracies, val_ious):
    """Plot training curves"""
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Loss curve
    axs[0, 0].plot(train_losses, label='Training', color=colors[0], linewidth=2)
    axs[0, 0].plot(val_losses, label='Validation', color=colors[1], linewidth=2)
    axs[0, 0].set_title('Training and Validation Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)

    # Accuracy curve
    axs[0, 1].plot(val_accuracies, label='Validation', color=colors[2], linewidth=2)
    axs[0, 1].set_title('Validation Accuracy')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Accuracy (%)')
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)

    # IoU curve
    axs[1, 0].plot(val_ious, label='Validation', color=colors[3], linewidth=2)
    axs[1, 0].set_title('Validation IoU')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('IoU')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)

    # Normalized metrics
    train_losses_np, val_losses_np = np.array(train_losses), np.array(val_losses)
    val_accuracies_np, val_ious_np = np.array(val_accuracies), np.array(val_ious)

    norm_train_loss = (train_losses_np - np.min(train_losses_np)) / (
                np.max(train_losses_np) - np.min(train_losses_np) + 1e-7)
    norm_val_loss = (val_losses_np - np.min(val_losses_np)) / (np.max(val_losses_np) - np.min(val_losses_np) + 1e-7)
    norm_val_acc = (val_accuracies_np - np.min(val_accuracies_np)) / (
                np.max(val_accuracies_np) - np.min(val_accuracies_np) + 1e-7)
    norm_val_iou = (val_ious_np - np.min(val_ious_np)) / (np.max(val_ious_np) - np.min(val_ious_np) + 1e-7)

    axs[1, 1].plot(norm_train_loss, label='Training Loss', color=colors[0], linewidth=2)
    axs[1, 1].plot(norm_val_loss, label='Validation Loss', color=colors[1], linewidth=2)
    axs[1, 1].plot(norm_val_acc, label='Validation Accuracy', color=colors[2], linewidth=2)
    axs[1, 1].plot(norm_val_iou, label='Validation IoU', color=colors[3], linewidth=2)
    axs[1, 1].set_title('Normalized Metrics')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Normalized Value')
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()


def save_training_metrics(train_losses, val_losses, val_accuracies, val_ious):
    """Save training metrics to JSON file"""
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_ious': val_ious
    }
    with open('training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train crack detection model')
    parser.add_argument('--train_img_dir', type=str, required=True, help='Training images directory')
    parser.add_argument('--train_mask_dir', type=str, required=True, help='Training masks directory')
    parser.add_argument('--val_img_dir', type=str, required=True, help='Validation images directory')
    parser.add_argument('--val_mask_dir', type=str, required=True, help='Validation masks directory')
    parser.add_argument('--model_save_dir', type=str, default='saved_models', help='Directory to save models')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()

    # Train model
    model = train_model(
        train_img_dir=args.train_img_dir,
        train_mask_dir=args.train_mask_dir,
        val_img_dir=args.val_img_dir,
        val_mask_dir=args.val_mask_dir,
        model_save_dir=args.model_save_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate
    )

    print("Training completed!")