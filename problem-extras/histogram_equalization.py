import numpy as np
import matplotlib.pyplot as plt


def make_gray_image(image):
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    return gray_image


def histogram_generate(image):
    histogram = np.zeros(256, dtype=int)
    for row in image:
        for gray_level in row:
            histogram[gray_level] += 1
    return histogram


def do_normalize_histogram(histogram, image):
    cdf = np.cumsum(histogram)
    norm_cdf = np.round((cdf/cdf[-1])*255).astype(np.uint8)

    normalized_image = norm_cdf[image]
    norm_histogram = np.zeros(256, dtype=int)
    for i in range(0, 256):
        norm_histogram[norm_cdf[i]] = histogram[i]
    return norm_histogram, normalized_image


rgb_image = plt.imread('images/lena.jpg')
gray_image = make_gray_image(rgb_image)

histogram = histogram_generate(gray_image)
normalized_histogram, normalized_image = do_normalize_histogram(histogram, gray_image)


plt.figure(figsize=(8, 7))
plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap='gray', vmin=0, vmax=255)
plt.title('Original')
plt.subplot(2, 2, 2)
plt.bar(range(256), histogram)
plt.title('Histogram')
plt.subplot(2, 2, 3)
plt.imshow(normalized_image, cmap='gray', vmin=0, vmax=255)
plt.title('Normalized Image')
plt.subplot(2, 2, 4)
plt.bar(range(256), normalized_histogram)
plt.title('Normalized Histogram')

plt.tight_layout()
plt.show()
