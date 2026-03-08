import os, torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image

BASE = "/kaggle/input/datasets/pkdarabi/bone-break-classification-image-dataset/Bone Break Classification/Bone Break Classification"
CLASSES = sorted(os.listdir(BASE))
C2I = {c:i for i,c in enumerate(CLASSES)}

class FracDS(Dataset):
    def _init_(self, split, tfm):
        self.data, self.tfm = [], tfm
        for c in CLASSES:
            p = os.path.join(BASE, c, split)
            if not os.path.exists(p): continue
            for f in os.listdir(p):
                if f.lower().endswith(('.jpg','.jpeg','.png')):
                    self.data.append((os.path.join(p,f), C2I[c]))
    def _len_(self): return len(self.data)
    def _getitem_(self, i):
        img, lbl = self.data[i]
        return self.tfm(Image.open(img).convert('RGB')), lbl

def get_loaders(batch_size=32):
    Ttr = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    Tte = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    tr = FracDS('Train', Ttr)
    te = FracDS('Test',  Tte)
    n  = int(0.9*len(tr))
    tr_ds, va_ds = random_split(tr, [n, len(tr)-n])
    return (
        DataLoader(tr_ds, batch_size, shuffle=True,  num_workers=2),
        DataLoader(va_ds, batch_size, shuffle=False, num_workers=2),
        DataLoader(te,    batch_size, shuffle=False, num_workers=2),
        FracDS('Train', Tte).data  # for class info
    )

if _name_ == "_main_":
    train_loader, val_loader, test_loader, _ = get_loaders()
    print("Classes:", CLASSES)
    print("✅ Data loader ready!")
