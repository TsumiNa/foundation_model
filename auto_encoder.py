import torch
import torch.nn as nn
import lightning.pytorch as pl

class Encoder(nn.Module):
    def __init__(self, input_dim=232, hidden_dims=[128, 64, 32]):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, output_dim=232, hidden_dims=[32, 64, 128]):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], output_dim),
        )
    
    def forward(self, x):
        return self.layers(x)

class EncoderDecoderModel(pl.LightningModule):
    def __init__(self, input_dim=232, hidden_dims=[128, 64, 32], *, lr=1e-3):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, hidden_dims=hidden_dims)
        self.decoder = Decoder(output_dim=input_dim, hidden_dims=hidden_dims[::-1])
        self._lr = lr
        
    def forward(self, x):
        # Get encoded representation
        encoded = self.encoder(x)
        # Reconstruct input
        decoded = self.decoder(encoded)
        return decoded
    
    def training_step(self, batch, batch_idx):
        x = batch
        # Forward pass
        reconstructed = self(x)
        # Calculate reconstruction loss
        loss = nn.functional.mse_loss(reconstructed, x)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        reconstructed = self(x)
        val_loss = nn.functional.mse_loss(reconstructed, x)
        self.log('val_loss', val_loss)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        return optimizer
