import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasetLoader import ImageTensorDataset, show_rgb_image
from ANNModel import ANNModel, ANNTrainer

# **DEVICE CONFIGURATION**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# **LOAD DATASET**
train_dataset = ImageTensorDataset("DATASET/TRAINING")
test_dataset = ImageTensorDataset("DATASET/TESTING")

# **BUAT DATALOADER**
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# **INISIALISI MODEL**
input_size = 224 * 224 * 3  # Flatten dari citra RGB ukuran 224x224
num_classes = len(train_dataset.class_to_idx)  # Jumlah kelas

idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}  # Mapping indeks ke nama folder

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

model = ANNModel(input_size, num_classes).to(device)
trainer = ANNTrainer(model, train_loader, test_loader, optimizer_type=optimizer_type)

# **TRAINING MODEL**
model.train()
epochs = 50
training_errors = trainer.train(epochs=epochs)

# **EVALUASI DENGAN DATA TRAINING**
model.eval()
train_results = []
correct = 0
total = 0
all_train_preds = []
all_train_labels = []
with torch.no_grad():
    for i in range(len(train_dataset)):
        img_tensor, label = train_dataset[i]
        img_tensor = img_tensor.to(device)
        img_tensor = img_tensor.reshape(-1, input_size)
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        pred_label = idx_to_class[predicted.item()]
        true_label = idx_to_class[label.item()]
        file_name = f"train_image_{i}.png"
        train_results.append([file_name, pred_label, true_label])
        total += 1
        correct += (predicted.item() == label.item())
        all_train_preds.append(predicted.item())
        all_train_labels.append(label.item())

train_accuracy = correct / total
train_error_rate = 1 - train_accuracy
print(f"Train Accuracy: {train_accuracy:.4f}, Train Error Rate: {train_error_rate:.4f}")

# **SIMPAN HASIL TRAINING KE CSV**
pd.DataFrame(train_results, columns=["File Name", "Predicted Label", "Actual Label"]).to_csv("train_results.csv", index=False)

# **TESTING MODEL**
accuracy, all_preds, all_labels = trainer.test()

test_results = []
with torch.no_grad():
    for i in range(len(test_dataset)):
        img_tensor, label = test_dataset[i]
        img_tensor = img_tensor.to(device)
        img_tensor = img_tensor.reshape(-1, input_size)
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        pred_label = idx_to_class[predicted.item()]
        true_label = idx_to_class[label.item()]
        file_name = f"test_image_{i}.png"
        test_results.append([file_name, pred_label, true_label])

# **SIMPAN HASIL TESTING KE CSV**
pd.DataFrame(test_results, columns=["File Name", "Predicted Label", "Actual Label"]).to_csv("test_results.csv", index=False)

# **EVALUASI MODEL DENGAN DATA TESTING**
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')

print(f"Evaluation Metrics:\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}")

# **PLOT ERROR TRAINING**
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), training_errors, marker='o', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Error over Epochs")
plt.show()

print("Hasil prediksi data training disimpan di train_results.csv")
print("Hasil prediksi data testing disimpan di test_results.csv")

# **MENAMPILKAN GAMBAR HASIL TESTING**
num_images = 8  # Jumlah gambar yang ingin ditampilkan
fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 baris, 5 kolom
axes = axes.flatten()

for i in range(num_images):
    img_tensor, label = test_dataset[i]
    img = img_tensor.permute(1, 2, 0).cpu().numpy()  # Konversi tensor ke numpy array (H, W, C)

    # Prediksi label
    img_tensor = img_tensor.to(device).reshape(-1, input_size)
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)

    pred_label = idx_to_class[predicted.item()]
    true_label = idx_to_class[label.item()]

    # Tampilkan gambar
    axes[i].imshow(img)
    axes[i].axis("off")
    axes[i].set_title(f"Actual: {true_label}\nPredicted: {pred_label}")

plt.tight_layout()
plt.show()

# Simpan model ke file
model_save_path = "trained_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model telah disimpan ke {model_save_path}")