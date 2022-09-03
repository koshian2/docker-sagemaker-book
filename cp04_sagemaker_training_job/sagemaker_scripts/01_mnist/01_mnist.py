import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, inputs):
        x = inputs.view(inputs.shape[0], 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    train_dataset = torchvision.datasets.MNIST(
        "./data", train=False, download=True,
        transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(
        "./data", train=False, download=True,
        transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, num_workers=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, num_workers=4, shuffle=False)

    model = MNISTModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()

            pred = model(inputs)
            loss = criterion(pred, labels)

            loss.backward()
            optimizer.step()

        model.eval()
        n_correct, n_total = 0, 0
        for inputs, labels in test_loader:
            pred = model(inputs)
            n_correct += torch.sum(pred.argmax(dim=-1) == labels)
            n_total += labels.shape[0]
        print(f"Epoch   {epoch} | TestAcc={n_correct/n_total:2%}")

if __name__ == "__main__":
    main()