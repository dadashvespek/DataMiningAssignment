import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import random
from blur import ImageBlurrer
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid

class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        self.blurrer = ImageBlurrer()
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        img_np = image.numpy().transpose(1, 2, 0)
        
        # technique of blurring picked at random
        blur_choice = random.choice(['average', 'gaussian', 'median'])
        if blur_choice == 'average':
            blurred = self.blurrer.average_blur(img_np, kernel_size=random.choice([3, 5, 7]))
        elif blur_choice == 'gaussian':
            blurred = self.blurrer.gaussian_blur(img_np, kernel_size=random.choice([3, 5, 7]), 
                                               sigma=random.uniform(0.5, 2.0))
        else:
            blurred = self.blurrer.median_blur(img_np, kernel_size=random.choice([3, 5]))
        blurred = torch.from_numpy(blurred.transpose(2, 0, 1)).float()
        
        return blurred, image

# U-Net style model for deblurring
class DeblurNet(nn.Module):
    def __init__(self):
        super(DeblurNet, self).__init__()
        
        # encoder
        self.enc1 = self._make_layer(3, 64)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        
        # decoder
        self.dec3 = self._make_layer(256, 128)
        self.dec2 = self._make_layer(128, 64)
        self.dec1 = self._make_layer(64, 32)
        
        # final
        self.final = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.max_pool2d(e1, 2))
        e3 = self.enc3(nn.functional.max_pool2d(e2, 2))
        d3 = self.dec3(nn.functional.interpolate(e3, scale_factor=2))
        d2 = self.dec2(nn.functional.interpolate(d3 + e2, scale_factor=2))
        d1 = self.dec1(d2 + e1)
        return self.sigmoid(self.final(d1))

def save_sample_images(model, test_images, epoch, save_dir='samples'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        blurred, original = test_images
        output = model(blurred)
        comparison = torch.cat([blurred, output, original], dim=0)
        save_image(comparison, f'{save_dir}/epoch_{epoch}.png', 
                  nrow=blurred.size(0), normalize=True)
    model.train()

def visualize_results(image_path, model, transform, device):
    image = Image.open(image_path).convert('RGB')
    original = transform(image).unsqueeze(0)
    blurrer = ImageBlurrer()
    img_np = original.squeeze(0).numpy().transpose(1, 2, 0)
    blurred = blurrer.gaussian_blur(img_np, kernel_size=5, sigma=1.0)
    blurred = torch.from_numpy(blurred.transpose(2, 0, 1)).float().unsqueeze(0)
    model.eval()
    with torch.no_grad():
        deblurred = model(blurred.to(device))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(blurred.squeeze(0).permute(1, 2, 0))
    ax1.set_title('Blurred')
    ax2.imshow(deblurred.cpu().squeeze(0).permute(1, 2, 0))
    ax2.set_title('Deblurred')
    ax3.imshow(original.squeeze(0).permute(1, 2, 0))
    ax3.set_title('Original')
    plt.show()

def train_model(model, train_loader, num_epochs, device, checkpoint_dir='checkpoints'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    vis_images = next(iter(train_loader))
    vis_images = [img.to(device) for img in vis_images]
    train_losses = []
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for blurred, target in progress_bar:
            blurred, target = blurred.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(blurred)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss/len(train_loader)
        train_losses.append(avg_loss)
        save_sample_images(model, vis_images, epoch + 1)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        torch.save(checkpoint, f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth')
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses)
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('training_loss.png')
        plt.close()

def load_checkpoint(model, checkpoint_path, device):
    """Load a specific checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['epoch']

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = DeblurDataset('DIV2K_train_HR', transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    model = DeblurNet().to(device)
    
    
    num_epochs = 20 
    train_model(model, dataloader, num_epochs=num_epochs, device=device)
    torch.save(model.state_dict(), 'deblur_model_final.pth')
    
    test_image_path = 'DIV2K_train_HR/001.png' 
    visualize_results(test_image_path, model, transform, device)

if __name__ == '__main__':
    main()