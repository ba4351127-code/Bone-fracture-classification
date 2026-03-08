import torch
import torch.nn as nn
import timm

NUM_CLASSES = 10

def get_vit():
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=False,
        num_classes=NUM_CLASSES
    )
    return model

def get_efficientnet():
    model = timm.create_model(
        'efficientnet_b4',
        pretrained=False,
        num_classes=NUM_CLASSES
    )
    return model

class EnsembleModel(nn.Module):
    def _init_(self):
        super()._init_()
        self.vit = get_vit()
        self.eff = get_efficientnet()
        self.fc  = nn.Linear(NUM_CLASSES * 2, NUM_CLASSES)

    def forward(self, x):
        v = self.vit(x)
        e = self.eff(x)
        return self.fc(torch.cat([v, e], dim=1))

if _name_ == "_main_":
    model = get_vit()
    print("✅ ViT Model ready!")
    print("Parameters:", sum(p.numel() for p in model.parameters()))
