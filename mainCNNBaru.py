import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasetLoader import ImageTensorDataset, show_rgb_image
from CNNBaru import CustomCNN

# **DEVICE CONFIGURATION**
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Model berjalan di: {device}")

# **LOAD DATASET**
train_dataset = ImageTensorDataset("DATASET/TRAINING")
test_dataset = ImageTensorDataset("DATASET/TESTING")

# **BUAT DATALOADER**
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# **INISIALISASI MODEL**
model = CustomCNN(num_classes=len(train_dataset.class_to_idx)).to(device)

# **PILIH METODE OPTIMASI**
print("Pilih metode optimasi:")
print("1: SGD")
print("2: Mini-Batch GD")
print("3: GD")
opt_choice = input("Masukkan pilihan (1/2/3): ")

optimizer_type = "SGD"
if opt_choice == "2":
    optimizer_type = "MiniBatchGD"
elif opt_choice == "3":
    optimizer_type = "GD"

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9 if optimizer_type == "MiniBatchGD" else 0)

# **TRAINING MODEL**
model.train()
epochs = 100
training_errors = []
for epoch in range(epochs):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    training_errors.append(avg_loss)
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

# **EVALUASI MODEL**
model.eval()
all_preds, all_labels = [], []
train_results, test_results = [], []
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

# **EVALUASI TRAINING DATASET**
with torch.no_grad():
    for i in range(len(train_dataset)):
        img_tensor, label = train_dataset[i]
        img_tensor = img_tensor.to(device).unsqueeze(0)
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        pred_label = idx_to_class.get(predicted.item(), "Unknown")
        true_label = idx_to_class.get(label.item(), "Unknown")
        file_name = f"train_image_{i}.png"
        train_results.append([file_name, pred_label, true_label])
        all_preds.append(predicted.item())
        all_labels.append(label.item())

# **SIMPAN HASIL TRAINING KE CSV**
pd.DataFrame(train_results, columns=["File Name", "Predicted Label", "Actual Label"]).to_csv("train_resultsCNN.csv",
                                                                                             index=False)

# **EVALUASI TESTING DATASET**
with torch.no_grad():
    for i in range(len(test_dataset)):
        img_tensor, label = test_dataset[i]
        img_tensor = img_tensor.to(device).unsqueeze(0)
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        pred_label = idx_to_class.get(predicted.item(), "Unknown")
        true_label = idx_to_class.get(label.item(), "Unknown")
        file_name = f"test_image_{i}.png"
        test_results.append([file_name, pred_label, true_label])
        all_preds.append(predicted.item())
        all_labels.append(label.item())

# **SIMPAN HASIL TESTING KE CSV**
pd.DataFrame(test_results, columns=["File Name", "Predicted Label", "Actual Label"]).to_csv("test_resultsCNN.csv",
                                                                                            index=False)

# **MENGHITUNG METRIK AKURASI**
if all_preds:
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
else:
    accuracy, precision, recall, f1 = 0, 0, 0, 0

print(
    f"Evaluation Metrics:\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}")

# **PLOT ERROR TRAINING**
if training_errors:
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), training_errors, marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Error over Epochs")
    plt.show()
else:
    print("No training loss recorded.")

print("Hasil prediksi data training disimpan di train_resultsCNN.csv")
print("Hasil prediksi data testing disimpan di test_resultsCNN.csv")
print("Model training dan evaluasi selesai.")

# **MENAMPILKAN GAMBAR HASIL TESTING DENGAN LABEL AKTUAL DAN PREDIKSI**
num_images = 8  # Jumlah gambar yang ingin ditampilkan
fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 baris, 5 kolom
axes = axes.flatten()

for i in range(num_images):
    img_tensor, label = test_dataset[i]
    img = img_tensor.permute(1, 2, 0).cpu().numpy()  # Konversi tensor ke numpy array (H, W, C)

    # Prediksi label
    img_tensor = img_tensor.to(device).unsqueeze(0)
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)

    pred_label = idx_to_class.get(predicted.item(), "Unknown")
    true_label = idx_to_class.get(label.item(), "Unknown")

    # Tampilkan gambar
    axes[i].imshow(img)
    axes[i].axis("off")
    axes[i].set_title(f"Actual: {true_label}\nPredicted: {pred_label}")

plt.tight_layout()
plt.show()