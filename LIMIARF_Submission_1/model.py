import timm
import torch
import torch.nn as nn

class DiamondModel(nn.Module):
    def __init__(self, args):
        super(DiamondModel, self).__init__()
        # Load ResNet-101 backbone from timm with custom number of classes
        self.backbone = timm.create_model(args.backbone, pretrained=False, num_classes=1000)  # Use 1000 as ResNet default
        
        # Load the pre-trained weights from the specified path
        state_dict = torch.load(args.weights_pth_file)
        state_dict = {k: v for k, v in state_dict.items() if "fc" not in k}
        self.backbone.load_state_dict(state_dict, strict=False)
        
        # Optionally freeze initial layers if desired
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Replace final fully connected layer (classifier)
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, args.num_classes)
        )

    def forward(self, x):
        # Forward pass through the backbone
        x = self.backbone(x)
        return x