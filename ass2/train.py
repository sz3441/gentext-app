import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from cnn import CIFARClassifier
from tqdm import tqdm
import os
import altair as alt
from mofresh import refresh_matplotlib, ImageRefreshWidget
import polars as pl
import matplotlib.pyplot as plt


device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print(f"Using device: {device}")


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

cifar_train = torchvision.datasets.CIFAR10(root = "data", train = True, download = True, transform = transform)
cifar_test = torchvision.datasets.CIFAR10(root = "data", train = False, download = True, transform = transform)

train_loader = DataLoader(cifar_train, batch_size = 32, shuffle = True)
test_loader = DataLoader(cifar_test, batch_size = 32, shuffle = False)



def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir = 'checkpoints_cnn'):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok = True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }

    checkpoint_path = os.path.join(checkpoint_dir, f'fcnn_epoch_{epoch:03d}.pth')
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location = device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']

    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

    return epoch, loss, accuracy



widget = ImageRefreshWidget(src = "")

@refresh_matplotlib
def losschart(data):
    df = pl.DataFrame(data)
    plt.plot(df["epoch"], df["train_loss"])
    plt.ylabel("Loss")
    plt.xlabel("Epoch")

widget


# training
EPOCHS = 15
model = CIFARClassifier().to(device)

datalogs = []
best_accuracy = 0.0

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0005)

for epoch in range(EPOCHS):
    running_loss = 0.0
    running_correct, running_total = 0, 0

    model.train()
    train_loader_with_progress = tqdm(iterable = train_loader, ncols = 120, desc = f'Epoch {epoch+1}/{EPOCHS}')
    for batch_number, (inputs, labels) in enumerate(train_loader_with_progress):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        # predicted = torch.argmax(outputs.data)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # log data for tracking
        running_correct += (predicted == labels).sum().item()
        running_total += labels.size(0)
        running_loss += loss.item()  

        if (batch_number % 100 == 99):
            train_loader_with_progress.set_postfix({'avg accuracy': f'{running_correct/running_total:.3f}', 'avg loss': f'{running_loss/(batch_number+1):.4f}'})

            datalogs.append({
                "epoch": epoch + batch_number / len(train_loader), 
                "train_loss": running_loss / (batch_number + 1),
                "train_accuracy": running_correct/running_total,
            })

    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * running_correct / running_total

    datalogs.append({
        "epoch": epoch + 1, 
        "train_loss": epoch_loss,
        "train_accuracy": running_correct/running_total,
    })

    # Save checkpoint every epoch
    checkpoint_path = save_checkpoint(
        model, optimizer, epoch + 1, epoch_loss, epoch_accuracy
    )

    # Save best model
    if epoch_accuracy > best_accuracy:
        best_accuracy = epoch_accuracy
        best_path = save_checkpoint(
            model, optimizer, epoch + 1, epoch_loss, epoch_accuracy, 
            checkpoint_dir='checkpoints_cnn/best'
        )
        print(f"New best model saved! Accuracy: {epoch_accuracy:.2f}%")

    print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.2f}%")
    print(f"Checkpoint saved: {checkpoint_path}")

    widget.src = losschart(datalogs)

print("Finished Training")





# test
test_correct = 0
test_total = 0
model.eval()
with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        test_outputs = model(test_images)
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_total += test_labels.size(0)
        test_correct += (test_predicted == test_labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print(f"Accuracy of the network on the 10000 test images: {test_accuracy:.2f}%")