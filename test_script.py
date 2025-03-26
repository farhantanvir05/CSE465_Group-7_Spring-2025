import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Read the CSV file
        with open(csv_file, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header if present
                parts = line.strip().split(",")

                # Ensure the line has at least two elements
                if len(parts) < 2:
                    print(f"Skipping invalid line: {line.strip()}")
                    continue

                image_path = parts[0].strip()
                label = parts[1].strip()

                # Ensure label is a valid integer
                if not label.isdigit():
                    print(f"Skipping line with invalid label: {line.strip()}")
                    continue

                label = int(label)

                if os.path.exists(image_path):  # Ensure file exists
                    self.image_paths.append(image_path)
                    self.labels.append(label)
                else:
                    print(f"Skipping missing file: {image_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros((3, 224, 224)), label  # Return dummy tensor if error occurs

csv_path = "/content/MURA-v1.1/augmented_image_paths.csv"
test_dataset = CustomImageDataset(csv_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Load the saved model
model = models.resnet50()
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)  # Adjust for binary classification

# Load model weights
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.to(device)
model.eval()

# Function to predict a single image
def predict_single_image(image_path):
    """Predicts the label of a single image using the trained model."""
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist.")
        return None

    image = Image.open(image_path).convert("RGB")  # Convert to RGB
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        output = model(image)
        _, prediction = torch.max(output, 1)

    print(f"Image: {image_path}, Predicted Label: {prediction.item()}")
    return prediction.item()

# Example: Predict a single image
image_path = "/content/MURA-v1.1/augmented/XR_HUMERUS/patient00180/study1_positive/image1.png"  # Change to your image path
predict_single_image(image_path)
