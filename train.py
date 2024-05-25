
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DatasetsAndLoaders(NamedTuple):
    train_dataset: datasets.MNIST
    test_dataset: datasets.MNIST
    train_loader: DataLoader
    test_loader: DataLoader

def get_datasets_and_loaders():
    # Normalize the pixel values to be between 0 and 1
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return DatasetsAndLoaders(train_dataset, test_dataset, train_loader, test_loader)

class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, test_loader, model_save_path, save_image_output_path):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_save_path = model_save_path
        self.save_image_output_path = save_image_output_path
    
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch+1}/{num_epochs}")
            for i, data in enumerate(self.train_loader):
                img, _ = data
                img = img.view(img.size(0), -1).to(device)
                
                # Forward pass
                output = self.model(img)
                loss = self.criterion(output, img)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                if i % 100 == 0:
                    print(f'Loss: {loss.item()}')
                self.optimizer.step()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        self.test()
        self.save_model()
        self.save_image_output()
    
    def test(self):
        with torch.no_grad():
            loss = 0
            for data in self.test_loader:
                img, _ = data
                img = img.view(img.size(0), -1).to(device)
                output = self.model(img)
                loss += self.criterion(output, img)
            print(f'Test Loss: {loss/len(self.test_loader)}')
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_save_path)
    
    def save_image_output(self):
        with torch.no_grad():
            dataiter = iter(self.test_loader)
            images, _ = next(dataiter)
            images = images.view(images.size(0), -1).to(device)
            
            outputs = self.model(images)
            
            # Plot the original and reconstructed images
            fig, axes = plt.subplots(2, 10, figsize=(10, 2))
            for i in range(10):
                # Original images
                ax = axes[0, i]
                ax.imshow(images[i].cpu().numpy().reshape(28, 28), cmap='gray')
                ax.axis('off')
                
                # Reconstructed images
                ax = axes[1, i]
                ax.imshow(outputs[i].cpu().numpy().reshape(28, 28), cmap='gray')
                ax.axis('off')

            # save the plot under images/
            plt.savefig(self.save_image_output_path)