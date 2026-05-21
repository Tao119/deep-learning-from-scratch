import sys
sys.path.append("..")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common.activation import step_function, sigmoid, relu


x = np.arange(-5.0, 5.0, 0.1)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].plot(x, step_function(x))
axes[0].set_title("Step Function")
axes[0].set_ylim(-0.1, 1.1)

axes[1].plot(x, sigmoid(x))
axes[1].set_title("Sigmoid")

axes[2].plot(x, relu(x))
axes[2].set_title("ReLU")

plt.tight_layout()
plt.savefig("activation_functions.png")
plt.close()
print("saved activation_functions.png")
