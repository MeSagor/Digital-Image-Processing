import numpy as np
import matplotlib.pyplot as plt
import cv2

def make_gray_image(image):
    gray_image = image
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = np.dot(image[..., :3], [0.30, 0.59, 0.11]).astype(np.uint8)
    return gray_image


def make_salt_pepper_noise(image):
    salt_pepper_probability = 0.17
    number_of_salt_pepper = np.ceil(salt_pepper_probability * image.size).astype(int)

    noisy_image = np.copy(image).astype(np.uint8)
    for i in range(number_of_salt_pepper):
        random_coordinate = np.random.randint(0, image.shape)
        if i & 1 == 0:
            noisy_image[random_coordinate[0], random_coordinate[1]] = 255
        else:
            noisy_image[random_coordinate[0], random_coordinate[1]] = 0
    return noisy_image


def calculate_window(x, y, kernel_size, image):
    window_left = kernel_size//2
    window_right = window_left+1
    height, width = image.shape

    window = []
    for i in range(x-window_left, x+window_right):
        row = []
        for j in range(y - window_left, y + window_right):
            row.append(image[i % height, j % width])
        window.append(np.array(row))
    return np.array(window)


def average_filter(image, kernel_size):
    average_mask = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    height, width = image.shape

    filter_image = np.zeros_like(image)
    for i in range(0, height):
        print(f'Row Processing[Average]: {i}', end='\r', flush=True)
        for j in range(0, width):
            average_value = np.sum(average_mask * calculate_window(i, j, kernel_size, image))
            filter_image[i, j] = average_value      
    return filter_image


def median_filter(image, kernel_size):
    median_mask = np.ones((kernel_size, kernel_size))
    height, width = image.shape

    filter_image = np.zeros_like(image)
    for i in range(0, height):
        print(f'Row Processing[Median]:  {i}', end='\r', flush=True)
        for j in range(0, width):
            median_value = np.median(median_mask * calculate_window(i, j, kernel_size, image))
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
gray_image = make_gray_image(rgb_image)
noisy_image = make_salt_pepper_noise(gray_image)

idx = 3
plt.figure(figsize=(10, 7))
for i in range(3, 10, 2):
    # filter_image = average_filter(noisy_image, kernel_size=i)
    filter_image = median_filter(noisy_image, kernel_size=i)
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
