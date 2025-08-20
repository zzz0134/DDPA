import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, shapiro, normaltest, anderson, probplot

# -------------------------
# Step 1: Load Tiny ImageNet
# -------------------------
transform = transforms.Compose([
    transforms.Resize(224),  # ResNet18 expects 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Assume tiny-imagenet-200 is extracted under ./data/tiny-imagenet-200
trainset = datasets.ImageFolder(root='/home/zzz0134/unlearning_attack/our_code/data/tiny-imagenet-200/train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# -------------------------
# Step 2: Load ResNet18
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 200)  # Tiny ImageNet has 200 classes
model = model.to(device)

# -------------------------
# Step 3: Train briefly (1 batch to update weights)
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print("Training ResNet18 on Tiny ImageNet for 1 batch...")
model.train()
for images, labels in trainloader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    break  #REMOVE this break if you want full 1 epoch

print("Training done.")

# -------------------------
# Step 4: Extract parameters from a layer
# -------------------------
layer_weights = model.conv1.weight.data.cpu().numpy().flatten()  # First Conv layer
print(f"Collected {len(layer_weights)} parameters from conv1")

# -------------------------
# Step 5: Histogram + Gaussian Fit
# -------------------------
mu, std = norm.fit(layer_weights)
plt.hist(layer_weights, bins=50, density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title(f"Histogram & Gaussian Fit: mu={mu:.2f}, std={std:.2f}")
plt.xlabel("Weight Value")
plt.ylabel("Density")
plt.grid(True)
plt.show()

# -------------------------
# Step 6: QQ plot
# -------------------------
probplot(layer_weights, dist="norm", plot=plt)
plt.title("QQ Plot of Layer Parameters")
plt.grid(True)
plt.show()

# -------------------------
# Step 7: Distribution Tests
# -------------------------
print("\n--- Normality Tests ---")

# Shapiro-Wilk Test
sample = layer_weights[:5000]  # Shapiro has limit
stat, p = shapiro(sample)
print(f"Shapiro-Wilk p-value: {p:.5f}")

# D’Agostino’s K² Test
stat, p = normaltest(layer_weights)
print(f"D’Agostino’s K² p-value: {p:.5f}")

# Anderson-Darling Test
result = anderson(layer_weights, dist='norm')
print(f"Anderson-Darling statistic: {result.statistic:.5f}")
print("Critical values:", result.critical_values)
print("Significance levels:", result.significance_level)
