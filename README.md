Alzheimer's Disease Detection using EfficientNet-B0
A Complete Guide: From Data Preparation to Deployment

This guide outlines the process of training a high-performance Deep Learning model (EfficientNet-B0) to detect Alzheimer's disease from MRI scans and deploying it as a live web application.

1. Environment Setup & Data Preparation
First, we mount Google Drive to access the dataset and copy it to the local Colab environment for faster training speeds.

```
from google.colab import drive
import os

# 1. Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# 2. Copy Dataset to Local Runtime (Faster I/O)
# Ensure these paths match your Drive structure
!cp -r "/content/drive/MyDrive/Colab Notebooks/Combined Dataset/train" /content/ 
!cp -r "/content/drive/MyDrive/Colab Notebooks/Combined Dataset/test" /content/

print(" Data successfully copied to Colab runtime!")
!ls /content/train | head -5
```


2. Define Custom Dataset Class
We define a custom PyTorch Dataset class to handle image loading, label encoding, and transformations

```
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision import transforms
import os

class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Auto-detect classes (alphabetical order)
        class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        
        print(f" Class Mapping for {root_dir}:")
        for k, v in self.class_to_idx.items():
            print(f"  • {k} → {v}")
            
        # Load images
        for class_name in class_names:
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_path, img_name))
                    self.labels.append(self.class_to_idx[class_name])
                    
        print(f" Loaded {len(self.images)} images.")

    def __len__(self): 
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform: 
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)
```
3. Data Augmentation & Loaders
We apply data augmentation techniques (rotation, flips, color jitter) to improve model generalization.
Prepares the data for training and testing the model.
 ```
# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Initialize Datasets & Loaders
train_dataset = MRIDataset('/content/train', transform=transform)
test_dataset = MRIDataset('/content/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("\n Data Loaders Ready!")
```
4. Initialize EfficientNet-B0 Model
We use Transfer Learning with timm to load a pre-trained EfficientNet-B0 model
```
 !pip install timm -q
import timm
import torch.nn as nn
import torch.optim as optim

# Load Pretrained EfficientNet-B0
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=4)

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)  # Low LR for fine-tuning

print(f" EfficientNet-B0 loaded on {device}! Ready for training.")
```
5. Model Training Loop
Train the model for 5 epochs and monitor the loss.
```
print("\n STARTING TRAINING (EfficientNet-B0 - 5 EPOCHS)...")

for epoch in range(5):
    model.train()
    running_loss = 0.0
    
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        if i % 100 == 0:
            print(f"  [Epoch {epoch+1}] Step {i}: Loss = {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f" Epoch {epoch+1} Complete → Avg Loss: {avg_loss:.4f}")

print("\n TRAINING COMPLETE!")
```
<img width="339" height="445" alt="image" src="https://github.com/user-attachments/assets/4947c79c-8e6f-4e02-8959-e7a559cab9b1" />

6. Evaluation & Saving
Evaluate the model on the test set and save the weights for deployment
Evaluates the trained EfficientNet-B0 model on the test dataset and calculates final accuracy
```
# 1. Evaluate Accuracy
model.eval()
correct = total = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

accuracy = 100 * correct / total
print(f"\n FINAL ACCURACY: {accuracy:.2f}%")
print(f"   Correct Predictions: {correct} / {total}")

# 2. Save Model
torch.save(model.state_dict(), '/content/efficientnet_b0_final.pth')
!mkdir -p "/content/drive/MyDrive/Alzheimer_MVP"
!cp /content/efficientnet_b0_final.pth "/content/drive/MyDrive/Alzheimer_MVP/"

print(" Model saved permanently to Google Drive!")
```
(Optional, but recomended for easiness)
```
torch.save(model.state_dict(), '/content/efficientnet_b0_final.pth')
 !mkdir -p "/content/drive/MyDrive/Alzheimer_MVP"
 !cp /content/efficientnet_b0_final.pth 
"/content/drive/MyDrive/Alzheimer_MVP/"
 print(" MODEL SAVED PERMANENTLY!")
```
7. Deployment (Streamlit + Ngrok)
Create a web interface and expose it publicly using Ngrok.

Step 7.1: Create the App File
```
%%writefile /content/app.py
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import timm

# --- Configuration ---
st.set_page_config(page_title="Alzheimer's AI", page_icon=")

# --- Model Loading ---
@st.cache_resource
def load_model():
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=4)
    model.load_state_dict(torch.load('/content/efficientnet_b0_final.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

# --- UI Layout ---
st.title(" Alzheimer's MRI Analysis AI")
st.markdown("### Powered by EfficientNet-B0 • 98%+ Accuracy")
st.write("Upload an MRI scan to detect early signs of Alzheimer's disease.")

uploaded = st.file_uploader("Upload MRI Image", type=['jpg', 'jpeg', 'png'])

if uploaded:
    col1, col2 = st.columns(2)
    
    img = Image.open(uploaded).convert('RGB')
    with col1:
        st.image(img, caption="Uploaded Scan", use_column_width=True)
    
    # Inference
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = torch.softmax(model(x), 1)[0]
        confidence, i = pred.max(0)
        
    diagnosis = classes[i.item()]
    
    with col2:
        st.success(f"**Diagnosis: {diagnosis}**")
        st.info(f"Confidence: {confidence.item()*100:.1f}%")
        
    st.markdown("---")
    st.markdown("#### Detailed Probability Distribution")
    st.bar_chart({c: float(pred[j]) for j, c in enumerate(classes)})
```
Step 7.2: Launch the Server
```
# 1. Install Dependencies
!pip install pyngrok streamlit -q

# 2. Authenticate Ngrok
from pyngrok import ngrok
import time

# REPLACE WITH YOUR TOKEN
NGROK_TOKEN = "PASTE_YOUR_NGROK_TOKEN_HERE" 
ngrok.set_auth_token(NGROK_TOKEN)

# 3. Clean up old processes
!pkill -f streamlit
!pkill -f ngrok

# 4. Run Streamlit in Background
!nohup streamlit run /content/app.py --server.port=8501 --server.headless=true > log.txt 2>&1 &

# 5. Create Tunnel
time.sleep(5)
try:
    tunnel = ngrok.connect(8501, bind_tls=True)
    print(f"\n LIVE DEMO READY!")
    print(f" Click here to open App: {tunnel.public_url}")
except Exception as e:
    print(f"Error starting tunnel: {e}")
 ```
