import os
import torch
import numpy as np
from model2 import UNet
from torchvision import transforms
from PIL import Image

# Function to load the model

def load_model(model_path, device):
    model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Function to preprocess the image

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to match model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to save the prediction

def save_prediction(prediction, save_path):
    prediction = prediction.squeeze().cpu().numpy()
    prediction = (prediction > 0.5).astype(np.uint8) * 255  # Threshold and convert to binary image
    Image.fromarray(prediction).save(save_path)

# Main function to segment images

def segment_images(model_path, image_dir, output_dir, device):
    model = load_model(model_path, device)
    os.makedirs(output_dir, exist_ok=True)

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = preprocess_image(image_path).to(device)

        with torch.no_grad():
            prediction = model(image)

        save_path = os.path.join(output_dir, f'{os.path.splitext(image_name)[0]}_segmented.png')
        save_prediction(prediction, save_path)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'checkpoints/best_model.pth'
    image_dir =  '/home/data/yanghong/data2/20201103苏芷琪-双侧卵巢浆液性交界性肿瘤/pic2'
    output_dir = 'segmented_results'
    segment_images(model_path, image_dir, output_dir, device)