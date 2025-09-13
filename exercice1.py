import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import signal

# Input utilisateurs que j'ai rajouté
taille_noyau = int(input("Entrez la taille du noyau de flou (ex: 3, 5, 7) : "))
seuil = int(input("Entrez le seuil de seuillage (0 à 255) : "))

noyauX = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

noyauY = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)

def generateur_noyau_flou(taille_noyau):
    noyau = (1 / taille_noyau ** 2) * np.ones((taille_noyau, taille_noyau), dtype=np.float32)
    return noyau


def convolution(image, noyau):
    return signal.convolve2d(image, noyau, mode="same", boundary="fill", fillvalue=255)


def amplitude_gradient(imageX, imageY):
    return np.sqrt(imageX ** 2 + imageY ** 2)


def seuillage(image, seuil):
    return (image < seuil)


# Charger l'image du beau Mayence
image = Image.open("mayence.jpg").convert("L")

# Traitement de l'image
noyau = generateur_noyau_flou(3)
image_floue = convolution(image, noyau)
convolX = convolution(image_floue, noyauX)
convolY = convolution(image_floue, noyauY)
grad = amplitude_gradient(convolX, convolY)
image_finale = seuillage(grad, 25)

# Affichage des images (je crois que j'aurais pu utiliser une loop mais bon..)
fig, axes = plt.subplots(3, 2, figsize=(10, 12))

axes[0, 0].imshow(image, cmap="gray")
axes[0, 0].set_title("1. Image originale")
axes[0, 0].axis("off")

axes[0, 1].imshow(image_floue, cmap="gray")
axes[0, 1].set_title("2. Image floutée (3x3)")
axes[0, 1].axis("off")

axes[1, 0].imshow(convolX, cmap="gray")
axes[1, 0].set_title("3. Sobel X")
axes[1, 0].axis("off")

axes[1, 1].imshow(convolY, cmap="gray")
axes[1, 1].set_title("4. Sobel Y")
axes[1, 1].axis("off")

axes[2, 0].imshow(grad, cmap="gray")
axes[2, 0].set_title("5. Amplitude du gradient")
axes[2, 0].axis("off")

axes[2, 1].imshow(image_finale, cmap="gray")
axes[2, 1].set_title("6. Après seuillage")
axes[2, 1].axis("off")

plt.tight_layout()
plt.show()
