from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transformasi: Resize, konversi ke tensor, dan normalisasi
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalisasi ke [-1, 1]
])

# Load dataset langsung dari folder
train_dataset = datasets.ImageFolder(root="DATASET/TRAINING", transform=transform)
test_dataset = datasets.ImageFolder(root="DATASET/TESTING", transform=transform)

# DataLoader untuk batching dan shuffle
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Contoh membaca satu batch
for images, labels in train_loader:
    print(f"Batch Image Shape: {images.shape}")  # [32, 3, 224, 224]
    print(f"Batch Label: {labels}")  # Tensor label
    break  # Hanya cetak 1 batch