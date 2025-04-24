import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder

import sys 
from PIL import Image
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
from torchvision.io import read_image

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score

class MyModel(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    
    self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.fc = nn.Linear(32 * 7 * 7, 10)  # 32 channels, 7x7 after pooling

  def forward(self, x):
      out = self.layer1(x)
      out = self.layer2(out)
      out = out.view(out.size(0), -1)  # flatten
      out = self.fc(out)
      return out


model = MyModel()

model.load_state_dict(torch.load("model.pt"))

model.eval()


def predict(image_location):
  im = read_image(image_location)
  plt.imshow(im.permute(1, 2, 0)/255, cmap='gray')

  # Load the image
  image = Image.open(image_location).convert('L')  # Convert to grayscale

  # Define the transformations (same as used during training)
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
  ])

  # Apply the transformations
  image_tensor = transform(image)

  # Add a batch dimension
  image_tensor = image_tensor.unsqueeze(0)

  # Make a prediction
  model.eval()  # Set the model to evaluation mode
  with torch.no_grad():
      prediction = model(image_tensor).argmax(dim=1)

  print("The model predicts " + str(prediction.item()))  # Print the predicted class

image = sys.argv[1]
predict(image)