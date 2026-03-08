import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import pandas as pd
from model import get_vit
from data_loader import FracDS, CLASSES, C2I
from torchvision import transforms
from torch.utils.data import random_split

# Config
EPOCHS     = 50
BATCH_SIZE = 32
LR         = 1e-4
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
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

tr    = FracDS('Train', Ttr)
te    = FracDS('Test',  Tte)
n     = int(0.9*len(tr))
tr_ds, va_ds = random_split(tr, [n, len(tr)-n])

train_loader = DataLoader(tr_ds, BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(va_ds, BATCH_SIZE, shuffle=False, num_workers=2)

# Model
model     = get_vit().to(DEVICE)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

history      = []
best_val_acc = 0

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss, train_correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

    # Validate
    model.eval()
    val_correct, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    train_acc = train_correct / len(tr_ds)
    val_acc   = val_correct   / len(va_ds)
    f1        = f1_score(all_labels, all_preds, average='macro')

    history.append({
        'epoch':          epoch+1,
        'train_accuracy': round(train_acc, 4),
        'val_accuracy':   round(val_acc,   4),
        'f1_score':       round(f1,        4),
        'train_loss':     round(train_loss/len(train_loader), 4),
        'learning_rate':  scheduler.get_last_lr()[0],
        'overfitting_gap':round(train_acc - val_acc, 4)
    })

    print(f"Epoch {epoch+1:02d} | Train: {train_acc:.3f} | Val: {val_acc:.3f} | F1: {f1:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print("  ✅ Best model saved!")

    scheduler.step()

pd.DataFrame(history).to_csv('training_history.csv', index=False)
print(f"\n🎉 Training complete! Best Val: {best_val_acc:.3f}")
