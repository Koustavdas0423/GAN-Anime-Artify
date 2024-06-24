import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import os

# Assuming you have defined the Generator class in model.py
from model import Generator

if __name__ == "__main__":
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms for data preprocessing and augmentation
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize images to match model input size
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # Randomly change the brightness, contrast, saturation and hue
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize image data
    ])

    # Load your dataset
    dataset = datasets.ImageFolder(root='C:/AnimeGANv2/samples/inputs', transform=transform)

    # Define data loader
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # Initialize the Generator network with pre-trained weights
    generator = Generator().to(device)
    generator.load_state_dict(torch.load('face_paint_512_v2_0.pt', map_location=device))  # Load pre-trained weights

    # Define loss function (e.g., Mean Squared Error)
    criterion = nn.MSELoss()

    # Define optimizer for the Generator
    optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        generator.train()  # Set generator to training mode
        for i, (images, _) in enumerate(tqdm(data_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]')):
            images = images.to(device)  # Move images to the device (GPU or CPU)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = generator(images)

            # Calculate loss
            loss = criterion(outputs, images)  # Using Mean Squared Error as loss

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Print loss (optional)
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(data_loader)}], Loss: {loss.item():.4f}")

    # Save fine-tuned Generator weights
    os.makedirs('models', exist_ok=True)  # Ensure the directory exists
    torch.save(generator.state_dict(), 'models/fine_tuned_generator.pth')
