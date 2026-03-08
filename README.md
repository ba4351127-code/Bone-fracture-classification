# Bone Fracture Classification
## Team Phew — IIT Mandi Hackathon 2026

### Team Members
- Tanuj Kumar Singh — Model Developer
- Divya Goyal — Data Engineer  
- Bhavna Agrawal — Evaluation & Explainability

### Problem Statement
Multi-class bone fracture classification using X-ray images (10 classes)

### Dataset
- Source: Kaggle — Bone Break Classification Image Dataset
- Classes: 10 fracture types
- Train: 890 | Val: 99 | Test: 140

### Model Architecture
- Vision Transformer (ViT-Base/16)
- EfficientNet-B4 (Ensemble)
- Optimizer: AdamW | LR: 1e-4
- Epochs: 50 | Batch: 32

### Project Structure
bone-fracture-classification/
├── data_loader.py   # Dataset loading & preprocessing
├── model.py         # ViT + EfficientNet architecture
├── train.py         # Training loop
├── evaluate.py      # Metrics & evaluation
└── README.md
### How to Run
```bash
pip install torch torchvision timm scikit-learn pandas
python train.py
python evaluate.py
Results
Training in progress...
*Ctrl+S — save karo!*

---

## Ab saari files ready hain! ✅
data_loader.py ✅
model.py       ✅
train.py       ✅
evaluate.py    ✅
README.md      ✅
