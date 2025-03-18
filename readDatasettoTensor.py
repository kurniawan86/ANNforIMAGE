import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

import matplotlib.pyplot as plt
import torch

def show_rgb_image(image_tensor):
    """
    Menampilkan satu gambar dalam warna RGB normal.

    Args:
        image_tensor (torch.Tensor): Tensor gambar dengan format [C, H, W].
    """
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("Input harus berupa torch.Tensor")

    # Pastikan tensor memiliki 3 channel (RGB)
    if image_tensor.shape[0] != 3:
        raise ValueError("Tensor harus memiliki 3 channel (RGB)")

    # Konversi dari [C, H, W] ke [H, W, C] agar bisa ditampilkan di matplotlib
    img_np = image_tensor.permute(1, 2, 0).numpy()

    # Pastikan nilai pixel dalam rentang [0,1] sebelum ditampilkan
    if img_np.min() < 0 or img_np.max() > 1:
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # Normalisasi manual jika perlu

    # Tampilkan gambar
    plt.imshow(img_np)
    plt.axis("off")
    plt.show()

class ImageTensorDataset(Dataset):
    """
    Custom PyTorch Dataset untuk membaca gambar dari folder TRAINING atau TESTING.
    """
    def __init__(self, dataset_path, size=(224, 224)):
        self.image_tensors = []
        self.labels = []
        self.class_to_idx = {}  # Mapping class ke indeks
        current_label = 0

        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),  # Konversi ke tensor [0,1]
            # transforms.Lambda(lambda x: 1 - x)  # Inversi ke [1,0]
        ])

        for person in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person)
            if not os.path.isdir(person_path):
                continue

            # Tambahkan label jika belum ada
            if person not in self.class_to_idx:
                self.class_to_idx[person] = current_label
                current_label += 1

            # Baca semua gambar dalam folder
            for img_file in os.listdir(person_path):
                if img_file.endswith(".png"):
                    img_path = os.path.join(person_path, img_file)
                    img = Image.open(img_path).convert("RGB")  # Baca gambar
                    img_tensor = transform(img)  # Resize & convert ke tensor
                    self.image_tensors.append(img_tensor)
                    self.labels.append(self.class_to_idx[person])  # Simpan label

        # Konversi list ke tensor
        self.image_tensors = torch.stack(self.image_tensors)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.image_tensors)

    def __getitem__(self, idx):
        return self.image_tensors[idx], self.labels[idx]


# **LOAD DATASET SECARA TERPISAH**
train_dataset = ImageTensorDataset("DATASET/TRAINING")
test_dataset = ImageTensorDataset("DATASET/TESTING")

# **BUAT DATALOADER UNTUK TRAIN & TEST**
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# **TAMPILKAN INFO DATASET**
print(f"Training Set: {len(train_dataset)} images")
print(f"Testing Set: {len(test_dataset)} images")

# Ambil satu batch dari training loader
images, labels = next(iter(train_loader))

# Tampilkan 8 gambar pertama
fig, axes = plt.subplots(1, 8, figsize=(20, 5))
for i in range(8):
    img_np = images[i].permute(1, 2, 0).numpy()
    axes[i].imshow(img_np)
    axes[i].axis("off")
    axes[i].set_title(f"Label: {labels[i].item()}")
plt.show()

torch.save((train_dataset.image_tensors, train_dataset.labels), "train_tensor.pth")
torch.save((test_dataset.image_tensors, test_dataset.labels), "test_tensor.pth")

train_tensors, train_labels = torch.load("train_tensor.pth")
test_tensors, test_labels = torch.load("test_tensor.pth")

# Buat DataLoader
train_loader = DataLoader(list(zip(train_tensors, train_labels)), batch_size=32, shuffle=True)
test_loader = DataLoader(list(zip(test_tensors, test_labels)), batch_size=32, shuffle=False)

# mengeluarkan nilai satu image tensor
# Ambil satu gambar dari dataset
image_tensor, label = train_dataset[0]  # Ambil gambar pertama

# Cetak shape tensor
print(f"Shape of image tensor: {image_tensor.shape}")  # [C, H, W]

# Cetak sebagian nilai pixel untuk melihat isi tensor
print(f"Pixel values:\n{image_tensor}")

# Ambil satu gambar dari dataset
image_tensor, label = train_dataset[0]

# Tampilkan gambar dengan warna normal
show_rgb_image(image_tensor)