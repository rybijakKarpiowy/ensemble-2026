
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics
import torch

class ResBlock(nn.Module):
    """A Residual Block with SiLU activation as defined in the paper."""
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU() # SiLU activation 

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.silu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = self.silu(x)
        return x

class ResUNet(nn.Module):
    """Core U-Net Architecture integrated with Residual Blocks."""
    def __init__(self, in_channels=3, out_channels=1):
        super(ResUNet, self).__init__()
        # Encoder
        self.enc1 = ResBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ResBlock(128, 256)
        
        # Decoder
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResBlock(256, 128) 
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResBlock(128, 64)
        
        # Output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.upconv2(b)
        d2 = torch.cat([e2, d2], dim=1) 
        d2 = self.dec2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat([e1, d1], dim=1) 
        d1 = self.dec1(d1)
        return self.final_conv(d1)

class LitECGModel(pl.LightningModule):
    """PyTorch Lightning wrapper for the model, loss, and optimizer."""
    def __init__(self, learning_rate=0.005):
        super().__init__()
        self.save_hyperparameters() # Saves LR to checkpoints automatically
        self.model = ResUNet()
        self.criterion = nn.BCEWithLogitsLoss() # BCE with logits loss 

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        # Lightning automatically logs this for you
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # ADAM optimizer as defined in the methodology 
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

