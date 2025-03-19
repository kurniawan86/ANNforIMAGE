import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ANNModel(nn.Module):
    """
    Model Artificial Neural Network (ANN) untuk klasifikasi citra.
    """

    def __init__(self, input_size, num_classes):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)  # Fully connected layer 1
        self.fc2 = nn.Linear(512, 256)  # Fully connected layer 2
        # self.fc3 = nn.Linear(256, 128)  # Fully connected layer 3
        self.fc3 = nn.Linear(256, num_classes)  # Output layer
        self.dropout = nn.Dropout(0.3)  # Regularisasi untuk menghindari overfitting

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input agar cocok dengan layer pertama
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc3(x))
        x = self.fc3(x)
        return x


class ANNTrainer:
    """
    Trainer untuk model ANN dengan pilihan optimasi SGD, Mini-batch GD, atau GD.
    """

    def __init__(self, model, train_loader, test_loader, criterion=nn.CrossEntropyLoss(), lr=0.01,
                 optimizer_type="SGD"):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion

        if optimizer_type == "SGD":
            self.optimizer = optim.SGD(model.parameters(), lr=lr)
        elif optimizer_type == "MiniBatchGD":
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_type == "GD":
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0)
        else:
            raise ValueError("Optimizer type not recognized")

    def train(self, epochs=10):
        """Melatih model ANN."""
        self.model.train()
        training_errors = []
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in self.train_loader:
                images = images.view(images.size(0), -1)  # Flatten image
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.train_loader)
            training_errors.append(avg_loss)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
        return training_errors

    def test(self):
        """Mengukur akurasi model pada dataset testing dan menampilkan prediksi gambar."""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.view(images.size(0), -1)  # Flatten image
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = 100 * correct / total
        print(f"Testing Accuracy: {accuracy:.2f}%")
        return accuracy, all_preds, all_labels
