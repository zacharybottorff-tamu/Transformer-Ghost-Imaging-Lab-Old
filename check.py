import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os

# Array of images
a = np.load('./zresult/SMILE_Ray_p15_0.npy') #32,32

# repeat for 10 images
for i in range(10):
  # put image i in figure
  plt.imshow(a[i])
  # turn off axis
  plt.axis('off')
  # display open figure
  plt.show()

# put image i=9 in figure
plt.imshow(a[i])
# turn off axis
plt.axis('off')
# display open figure
plt.show()