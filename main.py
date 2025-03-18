# from readDataset import get_image_files, show_images, convert_to_tensor_10
#
#
# def main():
#     datasetPath = "DATASET"
#     imageData = get_image_files(datasetPath)
#     tensorImage =convert_to_tensor_10(imageData)
#     show_images(imageData, size=(224, 224))  # Resize gambar ke ukuran 224x224
#
# if __name__ == "__main__":
#     main()

import torch
from PIL import Image
import os
from torchvision import datasets, transforms

def load_images_to_tensor(dataset_path, size=(224, 224)):
    """
    Membaca semua gambar dalam dataset dan mengonversinya menjadi satu tensor.
    """
    image_tensors = []
    labels = []
    class_to_idx = {}  # Mapping class ke indeks
    current_label = 0

    for category in ["TRAINING", "TESTING"]:
        category_path = os.path.join(dataset_path, category)
        if not os.path.exists(category_path):
            continue

        for person in os.listdir(category_path):
            person_path = os.path.join(category_path, person)
            if not os.path.isdir(person_path):
                continue

            # Tambahkan label
            if person not in class_to_idx:
                class_to_idx[person] = current_label
                current_label += 1

            # Baca semua gambar
            for img_file in os.listdir(person_path):
                if img_file.endswith(".png"):
                    img_path = os.path.join(person_path, img_file)
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(size)  # Resize
                    img_tensor = transforms.ToTensor()(img)  # Konversi ke tensor
                    image_tensors.append(img_tensor)
                    labels.append(class_to_idx[person])  # Tambahkan label

    return torch.stack(image_tensors), torch.tensor(labels)  # Kembalikan tensor dataset

# Load dataset ke dalam tensor
image_tensors, labels = load_images_to_tensor("DATASET")

print(f"Dataset Tensor Shape: {image_tensors.shape}")  # [N, 3, 224, 224]
print(f"Label Tensor Shape: {labels.shape}")  # [N]

# Simpan ke file agar tidak membaca ulang setiap kali training
torch.save((image_tensors, labels), "dataset_tensor.pth")