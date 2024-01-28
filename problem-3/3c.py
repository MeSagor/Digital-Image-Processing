import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio


def make_gray_image(image):
    gray_image = np.dot(rgb_image[..., :3], [
                        0.30, 0.59, 0.11]).astype(np.uint8)
    return gray_image


def make_salt_pepper_noise(image):
    salt_pepper_probability = 0.15
    num_salt_pepper = np.ceil(salt_pepper_probability * image.size).astype(int)

    noisy_image = np.copy(image).astype(np.uint8)
    for i in range(num_salt_pepper):
        rand_coord = np.random.randint(0, image.shape)
        if i & 1 == 0:
            noisy_image[rand_coord[0], rand_coord[1]] = 255
        else:
            noisy_image[rand_coord[0], rand_coord[1]] = 0

    return noisy_image


def average_filter(image, kernel_size):
    kernel_left = kernel_size//2
    kernel_right = kernel_left+1
    average_mask = np.ones((kernel_size, kernel_size))/(kernel_size * kernel_size)
    height, width = image.shape

    filter_image = np.zeros_like(image)
    for i in range(kernel_left, height-kernel_left):
        for j in range(kernel_left, width-kernel_left):
            average_value = np.sum(
                average_mask * image[i-kernel_left: i+kernel_right, j-kernel_left: j+kernel_right])
            filter_image[i, j] = average_value

    return filter_image


def harmonic_geometric_mean_filter(image, kernel_size):
    pad_width = kernel_size // 2
    fil_image = np.pad(image, pad_width, mode='constant')
    fil_image2 = np.pad(image, pad_width, mode='constant')

    for i in range(pad_width, fil_image.shape[0] - pad_width):
        print(f"Row Processing[median]: {i}", end="\r", flush=True)
        for j in range(pad_width, fil_image.shape[1] - pad_width):
            neighborhood = fil_image[i-pad_width : i+pad_width+1, j-pad_width : j+pad_width+1]
            neighborhood2 = fil_image2[i-pad_width : i+pad_width+1, j-pad_width : j+pad_width+1]

            neighborhood = neighborhood[neighborhood != 0]
            harmonic_value = neighborhood.size/np.sum(1/neighborhood)

            neighborhood2 = neighborhood2[neighborhood2>0]
            prod = np.prod(neighborhood2)
            geo = prod ** (1/ neighborhood2.size)

            fil_image[i, j] = harmonic_value
            fil_image2[i, j] = geo
    return fil_image[pad_width: -pad_width, pad_width:-pad_width], fil_image2[pad_width: -pad_width, pad_width:-pad_width]




def calculate_psnr(original_image, noisy_image):
    original_image = original_image.astype(np.float64)
    noisy_image = noisy_image.astype(np.float64)
    mean_square_error = np.mean((original_image - noisy_image) ** 2)
    max_pixel_value = 255
    psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mean_square_error)

    return psnr


rgb_image = plt.imread('images/cat.jpg')
gray_image = make_gray_image(rgb_image)
noisy_image = make_salt_pepper_noise(gray_image)

kernel_size = 3
filtered_image_harmonic, filtered_image_geometric = harmonic_geometric_mean_filter(noisy_image, kernel_size)

abc = peak_signal_noise_ratio(gray_image, noisy_image)
print(abc)
print(gray_image)
print(noisy_image)
noisy_image_psnr = calculate_psnr(gray_image, noisy_image)
print(gray_image)
print(noisy_image)
harmonic_filter_psnr = calculate_psnr(gray_image, filtered_image_harmonic)
geometric_filter_psnr = calculate_psnr(gray_image, filtered_image_geometric)

plt.figure(figsize=(9, 7))
plt.subplot(221)
plt.imshow(gray_image, cmap='gray', vmin=0, vmax=255)
plt.title('Original')
plt.subplot(222)
plt.imshow(noisy_image, cmap='gray', vmin=0, vmax=255)
plt.title('Noisy image')
plt.title(f'Noisy image - PSNR: {noisy_image_psnr:.2f} dB')
plt.subplot(223)
plt.imshow(filtered_image_harmonic, cmap='gray', vmin=0, vmax=255)
plt.title(f'Filtered (harmonic) PSNR: {harmonic_filter_psnr:.2f} dB')
plt.subplot(224)
plt.imshow(filtered_image_geometric, cmap='gray', vmin=0, vmax=255)
plt.title(f'Filtered (geometric) PSNR: {geometric_filter_psnr:.2f} dB')

plt.tight_layout()
plt.show()
