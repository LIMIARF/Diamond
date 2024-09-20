import timm
import torch
import torch.nn as nn

class DiamondModel(nn.Module):
    def __init__(self, args):
        super(DiamondModel, self).__init__()

        self.backbone = timm.create_model(args.backbone, pretrained=False, num_classes=1000)  # Use 1000 as ResNet default
        
        state_dict = torch.load(args.weights_pth_file)
        
        state_dict = {k: v for k, v in state_dict.items() if "fc" not in k}
        
        self.backbone.load_state_dict(state_dict, strict=False)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, args.num_classes)
        )
    def forward(self, x):
        # Forward pass through the backbone
        x = self.backbone(x)
        return x
