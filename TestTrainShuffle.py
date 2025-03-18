import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
from PIL import Image
import os

class ImageTensorDataset(Dataset):
    """
    Custom PyTorch Dataset untuk memuat gambar dalam bentuk tensor.
    """
    def __init__(self, dataset_path, size=(224, 224)):
        self.image_tensors = []
        self.labels = []
        self.class_to_idx = {}  # Mapping class ke indeks
        current_label = 0

        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),  # Konversi ke tensor
            transforms.Lambda(lambda x: 1 - x)  # Normalisasi ke [1,0]
        ])

        for category in ["TRAINING", "TESTING"]:
            category_path = os.path.join(dataset_path, category)
            if not os.path.exists(category_path):
                continue

            for person in os.listdir(category_path):
                person_path = os.path.join(category_path, person)
                if not os.path.isdir(person_path):
                    continue

                # Tambahkan label jika belum ada
                if person not in self.class_to_idx:
                    self.class_to_idx[person] = current_label
                    current_label += 1

                # Baca semua gambar
                for img_file in os.listdir(person_path):
                    if img_file.endswith(".png"):
                        img_path = os.path.join(person_path, img_file)
                        img = Image.open(img_path).convert("RGB")
                        img_tensor = transform(img)  # Resize & convert
                        self.image_tensors.append(img_tensor)
                        self.labels.append(self.class_to_idx[person])  # Simpan label

        # Konversi list ke tensor
        self.image_tensors = torch.stack(self.image_tensors)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.image_tensors)

    def __getitem__(self, idx):
        return self.image_tensors[idx], self.labels[idx]

# Load dataset ke dalam PyTorch Dataset
dataset = ImageTensorDataset("DATASET")

# Pisahkan menjadi training (80%) dan testing (20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Buat DataLoader untuk training dan testing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Total Dataset: {len(dataset)} images")
print(f"Training Set: {len(train_dataset)} images")
print(f"Testing Set: {len(test_dataset)} images")

import matplotlib.pyplot as plt

# Ambil satu batch dari training loader
images, labels = next(iter(train_loader))

# Tampilkan 8 gambar pertama
fig, axes = plt.subplots(1, 8, figsize=(20, 5))
for i in range(8):
    img_np = images[i].permute(1, 2, 0).numpy()  # Ubah format dari [C, H, W] ke [H, W, C]
    axes[i].imshow(img_np)
    axes[i].axis("off")
    axes[i].set_title(f"Label: {labels[i].item()}")

plt.show()

torch.save((dataset.image_tensors, dataset.labels), "dataset_tensor.pth")

image_tensors, labels = torch.load("dataset_tensor.pth")
train_size = int(0.8 * len(image_tensors))
test_size = len(image_tensors) - train_size

# Pisahkan data
train_tensors, test_tensors = image_tensors[:train_size], image_tensors[train_size:]
train_labels, test_labels = labels[:train_size], labels[train_size:]

# Buat DataLoader
train_loader = DataLoader(list(zip(train_tensors, train_labels)), batch_size=32, shuffle=True)
test_loader = DataLoader(list(zip(test_tensors, test_labels)), batch_size=32, shuffle=False)