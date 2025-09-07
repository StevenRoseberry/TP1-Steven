import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def convolution(original, noyau):
    # Je veut des pads de 1 parce que je work dans des 3x3
    x, y = original.shape
    # padding pour lignes et colonnes
    ## mettre constant_values à 255 pour imiter exemple du prof (padding blanc et non noir)
    padded = np.pad(original, ((1, 1), (1, 1)), mode='constant', constant_values=255)
    image_convol = np.zeros((y, x), dtype=np.float32)
    for i in range(x):
        for j in range(y):
            # prend une région de 3x3
            region = padded[i:i+3, j:j+3]
            # la ligne 18 est celle qui convolutionne l'image
            image_convol[i, j] = np.sum(region * noyau)
    return image_convol

def afficher(img0, img1, img2):
    fig, axes = plt.subplots(ncols=3)
    axes[0].imshow(img0, cmap=plt.get_cmap('gray'))
    axes[0].set_title("Image Originale")
    axes[1].imshow(img1, cmap=plt.get_cmap('gray'))
    axes[1].set_title("Ma convolution")
    axes[2].imshow(img2, cmap=plt.get_cmap('gray'))
    axes[2].set_title("Convolution SciPy")

    plt.show()


# Image 15x15 fond noir (valeur 0)
image_dessin = np.full((15, 15), 0, dtype=np.float32)

# On dessine un carré gris (valeur 128) au centre
image_dessin[3:12, 3:12] = 128

# On peut aussi ajouter une croix blanche (valeur 255) à l'intérieur du carré
image_dessin[7, 5:10] = 255
image_dessin[5:10, 7] = 255

# Noyau utilisé pour la convolution
noyau = 1 / 9 * np.ones((3, 3))

# Appliquer la convolution
sortie = convolution(image_dessin, noyau)

# Appliquer la convolution de Scipy.signal
sortie2 = signal.convolve2d(image_dessin, noyau, mode='same', boundary='fill', fillvalue=255)

# Afficher les trois images
afficher(image_dessin, sortie, sortie2)
