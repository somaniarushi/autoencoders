import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from train import DatasetsAndLoaders, Trainer, get_datasets_and_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Latent space
        self.mu = nn.Linear(128, 64)
        self.logvar = nn.Linear(128, 64)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
        )
    
    def reparameterize(self, mu, logvar):
        """
        This function samples from the latent space using the reparameterization trick

        :param mu: mean from the encoder's latent space
        :param logvar: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(device)
        return mu + eps*std
    
    def forward(self, x):
        x = self.encoder(x)
        mu, logvar = self.mu(x), self.logvar(x)
        z = self.reparameterize(mu, logvar) # sample from the latent space
        x = self.decoder(z)
        # put through sigmoid
        x = torch.sigmoid(x)
        assert torch.max(x) <= 1 and torch.min(x) >= 0, f"Max: {torch.max(x)} || Min: {torch.min(x)}"
        return x, mu, logvar
    
model = VariationalAutoEncoder().to(device)
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum') # Binary cross-entropy
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp()) # Kullback-Leibler divergence
    return reproduction_loss + KLD

optimizer = optim.Adam(model.parameters(), lr=1e-2)

datasets_and_loaders = get_datasets_and_loaders()

class VAETrainer(Trainer):
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
                # ensure the image is between 0 and 1 by normalizing
                img = img / 255
                assert torch.max(img) <= 1 and torch.min(img) >= 0, f"Max: {torch.max(img)}, Min: {torch.min(img)}"


                # Forward pass
                output, mu, logvar = self.model(img)
                loss = self.criterion(img, output, mu, logvar)
                
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
                output, mu, logvar = self.model(img)
                loss += self.criterion(output, img, mu, logvar)
            print(f'Test Loss: {loss.item()}')
    
    def save_image_output(self):
        with torch.no_grad():
            dataiter = iter(self.test_loader)
            images, _ = next(dataiter)
            images = images.view(images.size(0), -1).to(device)
            
            outputs, mus, logvars = self.model(images)
            
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

trainer = VAETrainer(
    model,
    optimizer, 
    loss_function,
    datasets_and_loaders.train_loader, 
    datasets_and_loaders.test_loader,
    'models/vae_model_mse_loss_no_norm_1e.pt',
    'images/vae_output_mse_loss_no_norm_1e.png'
)

trainer.train(20)
