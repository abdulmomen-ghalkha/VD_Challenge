#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from skimage import io, img_as_float32
from tqdm import tqdm


# Dataset Class
class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 1]
        image = img_as_float32(io.imread(img_name, as_gray=True))
        target_name = self.data.iloc[idx, 2]
        target = img_as_float32(io.imread(target_name))
        
        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image, target


# Simple CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        return x


# UNet Model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc_conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.up_conv = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = torch.relu(self.enc_conv1(x))
        x2 = torch.relu(self.enc_conv2(self.pool(x1)))
        bottleneck = torch.relu(self.bottleneck(self.pool(x2)))
        up = torch.relu(self.up_conv(bottleneck))
        dec = torch.relu(self.dec_conv(up))
        output = self.final_conv(dec)
        return output


# Utility function to save outputs
def save_images(images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for idx, img in enumerate(images):
        img = img.squeeze().cpu().numpy()
        Image.fromarray((img * 255).astype('uint8')).save(os.path.join(output_dir, f"output_{idx}.png"))


# Main function
def main(args):
    # Prepare dataset and dataloader
    dataset_dir = args.dataset


    # Get the list of common files
    files = os.listdir(dataset_dir)
    file_nums = list(set([file.split("_")[0] for file in files if file.endswith(".png")]))
    file_nums.sort()
    # Define the CSV file path
    csv_file_path = 'file_locations.csv'
    input_str = '_input.png'
    target_str = '_target.png'
    # Create and write to the CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow(['File', 'input_path', 'target_path'])
        # Write the file locations
        for file in file_nums:
            input_path = os.path.join(dataset_dir, file + input_str)
            target_path = os.path.join(dataset_dir, file + target_str)
            csvwriter.writerow([file, input_path, target_path])

    print(f'CSV file created at {csv_file_path}')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
    dataset = ImageDataset(csv_file=csv_file_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Model selection for three channels
    def create_model(arch):
        if arch == "cnn":
            return SimpleCNN()
        elif arch == "unet":
            return UNet()
        else:
            raise ValueError("Invalid model architecture. Choose 'cnn' or 'unet'.")

    models = {
        "channel_1": create_model(args.model).to(args.device),
        "channel_2": create_model(args.model).to(args.device),
        "channel_3": create_model(args.model).to(args.device),
    }

    if args.mode == "train":
        # Training mode
        criterion = nn.MSELoss()
        optimizers = {
            name: optim.Adam(model.parameters(), lr=0.001) for name, model in models.items()
        }

        num_epochs = 10
        for epoch in range(num_epochs):
            for channel, model in models.items():
                model.train()
                running_loss = 0.0
                optimizer = optimizers[channel]
                for images, targets in tqdm(dataloader, desc=f"{channel} - Epoch {epoch+1}/{num_epochs}"):
                    images, targets = images.to(args.device), targets.to(args.device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                print(f"{channel} - Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

                # Save model for each channel
                torch.save(model.state_dict(), f"{args.model}_{channel}_model.pth")
                print(f"{channel} model saved to {args.model}_{channel}_model.pth")

    elif args.mode == "test":
        # Testing mode
        outputs = {channel: [] for channel in models}
        for channel, model in models.items():
            model_path = f"{args.model}_{channel}_model.pth"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No saved model found for {channel}. Please train the model first.")
            model.load_state_dict(torch.load(model_path))
            model.eval()

            with torch.no_grad():
                for images, _ in tqdm(dataloader, desc=f"{channel} - Running Inference"):
                    images = images.to(args.device)
                    output = model(images)
                    outputs[channel].append(output)

        # Save outputs for each channel
        for channel, channel_outputs in outputs.items():
            save_images(channel_outputs, os.path.join(args.output_dir, channel))
            print(f"{channel} inference outputs saved to {os.path.join(args.output_dir, channel)}")

    else:
        raise ValueError("Invalid mode. Choose 'train' or 'test'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test with multiple models for separate channels")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset CSV file")
    parser.add_argument("--model", type=str, choices=["cnn", "unet"], required=True, help="Model type: 'cnn' or 'unet'")
    parser.add_argument("--output_dir", type=str, default="inference", help="Directory to save inference outputs")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="Mode: 'train' or 'test'")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    args = parser.parse_args()

    main(args)


'''

python your_script.py \
    --dataset path/to/dataset.csv \
    --model cnn \
    --mode train \
    --device cuda


python your_script.py \
    --dataset path/to/dataset.csv \
    --model cnn \
    --mode test \
    --output_dir path/to/output \
    --device cuda


'''

