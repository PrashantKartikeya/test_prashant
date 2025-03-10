# import torch
# import torchvision
# import torchvision.transforms as transforms
# import torch.optim as optim
# from torchvision.models import resnet18
# import torch.nn as nn
# import time

# # Check if MPS (Metal Performance Shaders) is available on Apple Silicon devices
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print("Using MPS (Apple Silicon GPU)")
# else:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

# # Data Preprocessing: Define transformations (normalize the data)
# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),   # Data augmentation: Random horizontal flip
#     transforms.RandomCrop(32, padding=4), # Data augmentation: Random cropping
#     transforms.ToTensor(),                # Convert images to PyTorch tensors
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize using CIFAR-10 mean and std
# ])

# train_start_time = time.time()

# # Load CIFAR-10 training and test datasets
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ]))
# testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

# # Model: Load pre-defined ResNet-18 model and modify the final layer for CIFAR-10 classification (10 classes)
# model = resnet18(pretrained=False)
# model.fc = nn.Linear(512, 10)  # Modify the final fully connected layer (for 10 classes in CIFAR-10)
# model = model.to(device)

# # Loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training Loop
# def train(model, trainloader, epochs):
#     model.train()  # Set the model to training mode
#     for epoch in range(epochs):
#         start_time = time.time()
#         running_loss = 0.0
#         correct = 0
#         total = 0
#         for i, data in enumerate(trainloader, 0):
#             # Get the inputs and labels, send to device (MPS or CPU/GPU)
#             inputs, labels = data[0].to(device), data[1].to(device)
            
#             # Zero the parameter gradients
#             optimizer.zero_grad()
            
#             # Forward pass
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
            
#             # Backward pass and optimization
#             loss.backward()
#             optimizer.step()
            
#             # Track accuracy
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
            
#             running_loss += loss.item()
#             if i % 100 == 99:  # Print loss every 100 mini-batches
#                 print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 100:.3f}')
#                 running_loss = 0.0
        
#         end_time = time.time()
#         accuracy = 100 * correct / total
#         print(f"Epoch {epoch+1} took {end_time - start_time:.2f} seconds, Accuracy: {accuracy:.2f}%")
#     train_end_time = time.time()
#     print(f"⏳ Total Execution Time: {train_end_time - train_start_time:.4f} seconds")
#     print('Finished Training')

# # Evaluation on the test set
# inf_start_time = time.time()
# def test(model, testloader):
#     model.eval()  # Set the model to evaluation mode
#     correct = 0
#     total = 0
#     with torch.no_grad():  # No need to calculate gradients during testing
#         for data in testloader:
#             images, labels = data[0].to(device), data[1].to(device)
#             outputs = model(images)
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
#     print(f'Accuracy on test images: {100 * correct / total:.2f}%')

# # Train the model and evaluate
# train(model, trainloader, epochs=10)
# test(model, testloader)
# inf_end_time = time.time()
# print(f"⏳ Total inference Time: {inf_end_time - inf_start_time:.4f} seconds")

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.models import resnet18
import torch.nn as nn
import time

# Check if MPS (Metal Performance Shaders) is available on Apple Silicon devices
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

# Data Preprocessing: Define transformations (normalize the data)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),   # Data augmentation: Random horizontal flip
    transforms.RandomCrop(32, padding=4), # Data augmentation: Random cropping
    transforms.ToTensor(),                # Convert images to PyTorch tensors
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize using CIFAR-10 mean and std
])

# Training Loop
def train(model, trainloader, epochs):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            # Get the inputs and labels, send to device (MPS or CPU/GPU)
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Track accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            running_loss += loss.item()
            if i % 100 == 99:  # Print loss every 100 mini-batches
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        end_time = time.time()
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1} took {end_time - start_time:.2f} seconds, Accuracy: {accuracy:.2f}%")
    train_end_time = time.time()
    print(f"⏳ Total Execution Time: {train_end_time - train_start_time:.4f} seconds")
    print('Finished Training')

# Evaluation on the test set
def test(model, testloader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No need to calculate gradients during testing
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    print(f'Accuracy on test images: {100 * correct / total:.2f}%')

if __name__ == '__main__':  # Wrap main logic inside this block for multiprocessing safety
    train_start_time = time.time()

    # Load CIFAR-10 training and test datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    # Model: Load pre-defined ResNet-18 model and modify the final layer for CIFAR-10 classification (10 classes)
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(512, 10)  # Modify the final fully connected layer (for 10 classes in CIFAR-10)
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model and evaluate
    train(model, trainloader, epochs=100)
    
    inf_start_time = time.time()
    test(model, testloader)
    inf_end_time = time.time()
    print(f"⏳ Total inference Time: {inf_end_time - inf_start_time:.4f} seconds")

