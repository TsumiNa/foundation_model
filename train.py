import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl
from model import EncoderDecoderModel

def main():
    # Create sample dataset
    # Replace this with your actual data
    train_size = 10000
    val_size = 2000
    input_dim = 232
    
    # Generate random data for demonstration
    train_data = torch.randn(train_size, input_dim)
    val_data = torch.randn(val_size, input_dim)
    
    # Create dataloaders
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=64, num_workers=0)
    
    # Initialize model
    model = EncoderDecoderModel(input_dim=input_dim, lr=0.0001)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='auto',  # Automatically detect if GPU is available
        devices=1,
        enable_progress_bar=True,
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
