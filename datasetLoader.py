import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

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
        ])

        for person in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person)
            if not os.path.isdir(person_path):
                continue

            if person not in self.class_to_idx:
                self.class_to_idx[person] = current_label
                current_label += 1

            for img_file in os.listdir(person_path):
                if img_file.endswith(".png"):
                    img_path = os.path.join(person_path, img_file)
                    img = Image.open(img_path).convert("RGB")  # Baca gambar
                    img_tensor = transform(img)  # Resize & convert ke tensor
                    self.image_tensors.append(img_tensor)
                    self.labels.append(self.class_to_idx[person])

        self.image_tensors = torch.stack(self.image_tensors)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.image_tensors)

    def __getitem__(self, idx):
        return self.image_tensors[idx], self.labels[idx]


# **Fungsi untuk Transformasi Gambar**
def transform_image(image_path):
    """
    Fungsi untuk melakukan transformasi gambar agar sesuai dengan model ANN.

    Args:
        image_path (str): Path menuju gambar yang akan diproses.

    Returns:
        torch.Tensor: Tensor gambar yang sudah di-preprocess.
    """

    # **Transformasi yang diterapkan pada gambar**
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize gambar ke 224x224
        transforms.ToTensor(),  # Konversi ke tensor PyTorch
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalisasi -1 ke 1
    ])

    # **Membuka gambar menggunakan PIL**
    image = Image.open(image_path).convert("RGB")  # Pastikan gambar dalam format RGB
    image_tensor = transform(image).unsqueeze(0)  # Tambahkan batch dimension

    return image_tensor

def show_rgb_image(image_tensor):
    """
    Menampilkan satu gambar dalam warna RGB normal.
    """
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("Input harus berupa torch.Tensor")

    if image_tensor.shape[0] != 3:
        raise ValueError("Tensor harus memiliki 3 channel (RGB)")

    img_np = image_tensor.permute(1, 2, 0).numpy()

    if img_np.min() < 0 or img_np.max() > 1:
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    plt.imshow(img_np)
    plt.axis("off")
    plt.show()