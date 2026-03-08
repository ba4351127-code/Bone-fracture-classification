import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score,
    classification_report, confusion_matrix
)
from model import get_vit
from data_loader import FracDS, CLASSES
from torchvision import transforms
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Tte = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_ds     = FracDS('Test', Tte)
test_loader = DataLoader(test_ds, 32, shuffle=False, num_workers=2)

# Load model
model = get_vit().to(DEVICE)
model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs   = imgs.to(DEVICE)
        outputs = model(imgs)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

# Metrics
acc = accuracy_score(all_labels, all_preds)
f1  = f1_score(all_labels, all_preds, average='macro')

print(f"Test Accuracy: {acc:.4f}")
print(f"F1 Score:      {f1:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASSES))

# Save results
results = []
for i, (pred, true) in enumerate(zip(all_preds, all_labels)):
    results.append({
        'image_id':       i,
        'true_label':     CLASSES[true],
        'predicted_label':CLASSES[pred],
        'correct':        pred == true
    })

pd.DataFrame(results).to_csv('final_results.csv', index=False)

perf = pd.DataFrame({
    'metric': ['accuracy', 'f1_macro'],
    'value':  [round(acc,4), round(f1,4)]
})
perf.to_csv('model_performance_analysis.csv', index=False)

print("\n✅ Results saved!")
