import numpy as np
import matplotlib.pyplot as plt
import cv2

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
    average_mask = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    height, width = image.shape

    filter_image = np.zeros_like(image)
    for i in range(kernel_left, height-kernel_left):
        for j in range(kernel_left, width-kernel_left):
            average_value = np.sum(
                average_mask * image[i-kernel_left: i+kernel_right, j-kernel_left: j+kernel_right])
            filter_image[i, j] = average_value

    return filter_image


def median_filter(image, kernel_size):
    kernel_left = kernel_size//2
    kernel_right = kernel_left+1
    median_mask = np.ones((kernel_size, kernel_size))
    height, width = image.shape

    filter_image = np.zeros_like(image)
    for i in range(kernel_left, height-kernel_left):
        for j in range(kernel_left, width-kernel_left):
            median_value = np.median(
                median_mask * image[i-kernel_left: i+kernel_right, j-kernel_left: j+kernel_right])
            filter_image[i, j] = median_value

    return filter_image


def calculate_psnr(original_image, noisy_image):
    original_image = original_image.astype(np.float64)
    noisy_image = noisy_image.astype(np.float64)
    mean_square_error = np.mean((original_image - noisy_image) ** 2)
    max_pixel_value = 255
    psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mean_square_error)

    return psnr


rgb_image = plt.imread('images/lena.jpg')
# rgb_image = cv2.imread("images/lena.jpg")
# rgb_image = cv2.cvtColor(rgb_image,cv2.COLOR_BGR2GRAY)
# rgb_image = cv2.resize()

gray_image = make_gray_image(rgb_image)
noisy_image = make_salt_pepper_noise(gray_image)

idx = 3
plt.figure(figsize=(10, 7))
for i in range(3, 10, 2):
    # filter_image = average_filter(noisy_image, i)
    filter_image = median_filter(noisy_image, i)
    psnr = calculate_psnr(gray_image, filter_image)
    plt.subplot(2, 3, idx)
    plt.imshow(filter_image, cmap='gray', vmin=0, vmax=255)
    plt.title(f'filtered: ({i}x{i}) PSNR: {psnr:.2f} dB')
    idx += 1

noisy_image_psnr = calculate_psnr(gray_image, noisy_image)

plt.subplot(231)
plt.imshow(gray_image, cmap='gray', vmin=0, vmax=255)
plt.title('Original')
plt.subplot(232)
plt.imshow(noisy_image, cmap='gray', vmin=0, vmax=255)
plt.title(f'Noisy image - PSNR: {noisy_image_psnr:.2f} dB')

plt.tight_layout()
plt.show()
