import sys

import numpy as np
from PIL import Image

from persistence import load_network

def image_to_training_input(path_to_image):
    image = Image.open(path_to_image).convert("L")  # grayscale
    image = image.resize((28, 28))
    pixels = np.array(image, dtype=np.float64)

    # MNIST digits are white on black. Invert if the image looks black-on-white
    if pixels.mean() > 127:
        pixels = 255 - pixels

    pixels /= 255.0
    return pixels.reshape(784, 1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path-to-image>")
        sys.exit(1)

    net = load_network("models/model.json")
    x = image_to_training_input(sys.argv[1])
    output = net.feedforward(x)

    print(f"Predicted digit: {np.argmax(output)}")
    print(f"Confidence per digit: {[round(float(p), 3) for p in output.flatten()]}")
