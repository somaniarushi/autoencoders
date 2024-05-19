
import torch
import torch.nn as nn
import torch.optim as optim
from train import DatasetsAndLoaders, Trainer, get_datasets_and_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

datasets_and_loaders = get_datasets_and_loaders()
trainer = Trainer(
    model,
    optimizer, 
    criterion, 
    datasets_and_loaders.train_loader, 
    datasets_and_loaders.test_loader,
    'models/ae_model.pt',
    'images/ae_output.png'
)
trainer.train(1)