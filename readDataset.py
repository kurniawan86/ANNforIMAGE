
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pathlib import Path

from torchvision.transforms.v2.functional import normalize_image


def get_image_files(dataset_path):
    data = {"TRAINING": {}, "TESTING": {}}

    for category in ["TRAINING", "TESTING"]:
        category_path = Path(dataset_path) / category
        if not category_path.exists():
            continue

        for person_folder in category_path.iterdir():
            if person_folder.is_dir():
                image_files = sorted(str(file) for file in person_folder.glob("*.png"))
                data[category][person_folder.name] = image_files

    return data

def convert_image_size(img, size=(224, 224)):
    return img.resize(size)

def show_images(image_data, size=(224, 224)):
    for category, persons in image_data.items():
        for person, files in persons.items():
            num_images = len(files)
            cols = min(num_images, 4)  # Maksimal 4 gambar per baris
            rows = (num_images // cols) + (1 if num_images % cols else 0)

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
            fig.suptitle(f"{category} - {person}", fontsize=14)

            if rows == 1 and cols == 1:
                axes = [[axes]]  # Handle jika hanya 1 gambar
            elif rows == 1:
                axes = [axes]  # Handle jika hanya 1 baris

            for idx, file in enumerate(files):
                row, col = divmod(idx, cols)
                img = normalize_image(file, size)

                axes[row][col].imshow(img)
                axes[row][col].axis("off")
                axes[row][col].set_title(f"Img {idx+1}")

            # Hide empty subplots
            for idx in range(num_images, rows * cols):
                row, col = divmod(idx, cols)
                axes[row][col].axis("off")

            plt.tight_layout()
            plt.show()

def convert_to_tensor_10(img):
    """
    Mengonversi gambar (hasil dari convert_image_size) menjadi tensor PyTorch dengan nilai dalam rentang [1,0].

    Args:
        img (PIL.Image): Objek gambar yang telah diresize.

    Returns:
        torch.Tensor: Tensor gambar yang sudah dinormalisasi ke [1,0].
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # Normalisasi otomatis ke [0,1]
        transforms.Lambda(lambda x: 1 - x)  # Inversi untuk mendapatkan [1,0]
    ])

    return transform(img)  # Output: [C, H, W]


def show_tensor_images(image_tensors, max_images=4):
    num_images = len(image_tensors)
    cols = min(num_images, max_images)  # Maksimal 4 gambar per baris
    rows = (num_images // cols) + (1 if num_images % cols else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle("Normalized Images [1,0]", fontsize=14)

    if rows == 1 and cols == 1:
        axes = [[axes]]  # Handle jika hanya ada 1 gambar
    elif rows == 1:
        axes = [axes]  # Handle jika hanya 1 baris

    for idx, img_tensor in enumerate(image_tensors):
        row, col = divmod(idx, cols)

        img_np = img_tensor.permute(1, 2, 0).numpy()  # Ubah format [C, H, W] ke [H, W, C]
        axes[row][col].imshow(img_np)
        axes[row][col].axis("off")
        axes[row][col].set_title(f"Img {idx + 1}")

    # Hide empty subplots
    for idx in range(num_images, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].axis("off")

    plt.tight_layout()
    plt.show()