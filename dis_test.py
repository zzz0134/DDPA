import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, shapiro, normaltest, anderson, probplot

# -------------------------
# Step 1: Load CIFAR-100
# -------------------------
transform = transforms.Compose([
    transforms.Resize(224),  # VGG expects 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# -------------------------
# Step 2: Load VGG16
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096, 100)  # CIFAR-100 has 100 classes
model = model.to(device)

# -------------------------
# Step 3: Train briefly (1 epoch just to get weights updated)
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print("Training VGG16 on CIFAR-100 for 1 epoch...")
model.train()
for images, labels in trainloader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    break  

print("Training done.")

# -------------------------
# Step 4: Extract parameters from a layer
# -------------------------
layer_weights = model.features[0].weight.data.cpu().numpy().flatten()  # First Conv layer
print(f"Collected {len(layer_weights)} parameters from features[0]")

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
