"""FULL DATASET TESTING"""

import torch
from torchvision import transforms
from PIL import Image
import os
from glob import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNTransformerModel().to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

base_dir = "/content/MURA-v1.1/valid/XR_WRIST"

for patient_folder in sorted(os.listdir(base_dir)):
    patient_path = os.path.join(base_dir, patient_folder)
    if not os.path.isdir(patient_path):
        continue

    study_folders = [os.path.join(patient_path, d) for d in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, d))]
    patient_preds = []

    for study in study_folders:
        image_paths = glob(os.path.join(study, "*.png"))
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)
                prob = torch.sigmoid(output).item()
                patient_preds.append(prob)

    # Average all predictions for this patient
    if patient_preds:
        avg_prob = sum(patient_preds) / len(patient_preds)
        prediction = "Fracture: Yes" if avg_prob > 0.5 else "Fracture: No"
        print(f"{patient_folder} => {prediction} (Avg Prob: {avg_prob:.4f})")

"""SINGLE IMAGE TESTING"""

from PIL import Image
import torchvision.transforms as transforms

image_path = "/content/image2_1047_png.rf.6d8753139f1f13e21f64d385b3b78865.jpg"
image = Image.open(image_path).convert("RGB")
transformed = val_transform(image).unsqueeze(0).to(device)

model.load_state_dict(torch.load("best_model.pth"))
model.eval()

with torch.no_grad():
    output = model(transformed)
    prob = torch.sigmoid(output).item()
    pred = 1 if prob > 0.5 else 0

print(f"\n🖼 Prediction for '{image_path}':")
print(f"➡️  Probability of fracture: {prob:.4f}")
print(f"➡️  Predicted Label: {'Fractured (Abnormal)' if pred == 1 else 'Normal'}")