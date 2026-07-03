import torch
import torch.nn as nn
import torchvision.models as models
import timm

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=512):
        super().__init__()
        # 2-Layer MLP Projection Head as specified in the dossier
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        x = self.mlp(x)
        # L2 normalize the output to project onto the unit hypersphere
        return nn.functional.normalize(x, p=2, dim=1)

class SAREncoder(nn.Module):
    def __init__(self, projection_dim=512):
        super().__init__()
        # Load a pretrained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Modify the first convolutional layer to accept 2 channels (VV and VH)
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Copy weights from the first two channels of the pretrained model to give a head start
        with torch.no_grad():
            new_conv.weight[:, :2, :, :] = old_conv.weight[:, :2, :, :]
            
        resnet.conv1 = new_conv
        
        # Remove the classification head (fc layer)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # ResNet50 outputs 2048-dimensional features before the FC layer
        self.projection = ProjectionHead(input_dim=2048, output_dim=projection_dim)

    def forward(self, x):
        # x is expected to be [Batch, 2, Height, Width]
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.projection(x)

class OpticalEncoder(nn.Module):
    def __init__(self, projection_dim=512):
        super().__init__()
        # We use timm to easily load a ViT-Base model.
        # num_classes=0 removes the classification head, returning the raw 768-dim features
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        
        # ViT-Base outputs 768-dimensional features
        self.projection = ProjectionHead(input_dim=768, output_dim=projection_dim)

    def forward(self, x):
        # x is expected to be [Batch, 3, Height, Width] (Assuming RGB optical extraction)
        x = self.backbone(x)
        return self.projection(x)

class CrossModalRetrievalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sar_encoder = SAREncoder()
        self.opt_encoder = OpticalEncoder()

    def forward(self, sar_imgs, opt_imgs):
        sar_embeddings = self.sar_encoder(sar_imgs)
        opt_embeddings = self.opt_encoder(opt_imgs)
        return sar_embeddings, opt_embeddings

if __name__ == "__main__":
    # A quick dry-run test
    print("Initializing models...")
    model = CrossModalRetrievalModel()
    
    # Create fake dummy images (Batch of 4) to verify the dimensions
    dummy_sar = torch.randn(4, 2, 224, 224)
    dummy_opt = torch.randn(4, 3, 224, 224)
    
    sar_out, opt_out = model(dummy_sar, dummy_opt)
    
    print(f"SAR Embedding Shape: {sar_out.shape}")  # Should be [4, 512]
    print(f"Optical Embedding Shape: {opt_out.shape}")  # Should be [4, 512]
    print("Dual Encoder script loaded and tested successfully!")