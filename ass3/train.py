import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from gan import Discriminator, Generator
from tqdm import tqdm
import os
from mofresh import refresh_matplotlib, ImageRefreshWidget
import polars as pl
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image


device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print(f"Using device: {device}")





transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

mnist_train = torchvision.datasets.MNIST(root = "data", train = True, download = True, transform = transform)
train_loader = DataLoader(mnist_train, batch_size = 128, shuffle = True)




z_dim = 100
lr = 0.0002
epochs = 1

criterion = nn.BCEWithLogitsLoss()

gen = Generator(z_dim).to(device)
discr = Discriminator().to(device)

opt_gen = optim.Adam(gen.parameters(), lr = lr, betas = (0.5, 0.999))
opt_discr = optim.Adam(discr.parameters(), lr = lr, betas = (0.5, 0.999))

fixed_noise = torch.randn(64, z_dim).to(device)




os.makedirs("generated_images", exist_ok = True)
os.makedirs("checkpoints", exist_ok = True)




widget = ImageRefreshWidget(src="")
image_widget = ImageRefreshWidget(src="")

@refresh_matplotlib
def losschart(data):
    df = pl.DataFrame(data)
    plt.plot(df["epoch"], df["D loss"], label = "Discriminator Loss")
    plt.plot(df["epoch"], df["G loss"], label = "Generator Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()

@refresh_matplotlib
def imagechart(grid, epoch):
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(f"Epoch {epoch}")
    plt.axis("off")






# train gan:
datalogs = []

for epoch in range(epochs):
    train_loader_with_progress = tqdm(
        iterable = train_loader, ncols = 120, desc = f"Epoch {epoch+1}/{epochs}"
    )
    for batch_number, (real, _) in enumerate(train_loader_with_progress):
        real = real.to(device)
        batch_size = real.size(0)

        ## === Train Discriminator === ##
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        discr_real = discr(real)
        loss_real = criterion(discr_real, real_labels)

        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)

        discr_fake = discr(fake.detach())
        loss_fake = criterion(discr_fake, fake_labels)

        loss_discr = (loss_real + loss_fake) / 2

        opt_discr.zero_grad()
        loss_discr.backward()
        opt_discr.step()

        ## === Train Generator === ##
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)

        discr_fake = discr(fake)
        loss_gen = criterion(discr_fake, real_labels)

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()


        if batch_number % 100 == 0:
            train_loader_with_progress.set_postfix(
                {
                    "Batch": f"{batch_number}/{len(train_loader)}",
                    "D loss": f"{loss_discr.item():.4f}",
                    "G loss": f"{loss_gen.item():.4f}",
                }
            )
            datalogs.append(
                {
                    "epoch": epoch + batch_number / len(train_loader),
                    "Batch": batch_number/len(train_loader),
                    "D loss": loss_discr.item(),
                    "G loss": loss_gen.item(),
                }
            )
            widget.src = losschart(datalogs)

            with torch.no_grad():
                sample_fake = gen(fixed_noise).detach().cpu()

            grid = make_grid(sample_fake, nrow = 8, normalize = True)
            image_widget.src = imagechart(grid, epoch + 1)
            
    gen.eval()

    with torch.no_grad():
        fake = gen(fixed_noise).cpu()

    gen.train()

    save_image(
        fake,
        f"generated_images/epoch_{epoch + 1}.png",
        normalize = True,
        nrow = 8
    ) 

    torch.save(
        gen.state_dict(),
        f"checkpoints/generator_epoch_{epoch + 1}.pth"
    )

    torch.save(
        discr.state_dict(),
        f"checkpoints/discriminator_epoch_{epoch + 1}.pth"
    )





torch.save(
        gen.state_dict(),
        "checkpoints/generator_final.pth"
    )

print("Final generator saved")