import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from data.datasets import get_dataloader
from networks.resnet import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} in HPC")

TRAIN_DIR = "images/train"
VAL_DIR = "images/val"
BATCH_SIZE = 256
LEARNING_RATE = 0.0002
NUM_EPOCHS = 50
MODEL_SAVE_PATH = "models"


def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses,
             marker='o', label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses,
             marker='o', label='Val Loss')
    plt.title("Training and Validation Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("models/loss_curve.png")
    plt.show()
    print("Loss curve saved to models/loss_curve.png")


def train():
    train_loader, _ = get_dataloader(
        TRAIN_DIR, batch_size=BATCH_SIZE, shuffle=True)
    val_loader, _ = get_dataloader(
        VAL_DIR, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model()
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            images = images.to(device)
            labels = labels.to(device)

            labels = labels.float().unsqueeze(1)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                images = images.to(device)
                labels = labels.to(device)

                labels = labels.float().unsqueeze(1)
                output = model(images)
                loss = criterion(output, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"models/checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

    plot_loss(train_losses, val_losses)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
