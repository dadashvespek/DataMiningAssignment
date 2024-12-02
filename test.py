import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from blur import ImageBlurrer
import numpy as np
from deblur_model import DeblurNet  

def test_deblurring(image_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DeblurNet().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    original = transform(image).unsqueeze(0)
    blurrer = ImageBlurrer()
    img_np = original.squeeze(0).numpy().transpose(1, 2, 0)
    blurred = blurrer.gaussian_blur(img_np, kernel_size=5, sigma=1.0)
    blurred_tensor = torch.from_numpy(blurred.transpose(2, 0, 1)).float().unsqueeze(0)
    
    with torch.no_grad():
        deblurred = model(blurred_tensor.to(device))
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(blurred)
    ax1.set_title('Blurred')
    ax1.axis('off')
    
    deblurred_img = deblurred.cpu().squeeze(0).permute(1, 2, 0).numpy()
    ax2.imshow(deblurred_img)
    ax2.set_title('Deblurred')
    ax2.axis('off')
    
    original_img = original.squeeze(0).permute(1, 2, 0).numpy()
    ax3.imshow(original_img)
    ax3.set_title('Original')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    plt.close()
if __name__ == '__main__':
    image_path = '0348.png'  
    model_path = 'deblur_model_final.pth' 
    
    test_deblurring(image_path, model_path)