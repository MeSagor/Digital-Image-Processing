import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from mpl_toolkits.mplot3d import Axes3D

def make_gray_image(image):
    gray_image = image
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = np.dot(image[..., :3], [0.30, 0.59, 0.11]).astype(np.uint8)
    return gray_image

def add_gaussian_noise(image, mean=0, std=25):
    noisy_image = image.copy()
    height, width = image.shape
    noise = np.random.normal(mean, std, (height, width))
    noisy_image = noisy_image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image

def frequency_bin_index(img_size=31, sigma=1.5):
   kernel = np.zeros((img_size, img_size))
   center = img_size // 2
   for u in range(img_size):
       for v in range(img_size):
           x, y = u - center, v - center
           kernel[u, v] = np.sqrt(x**2 + y**2)
   return kernel

def ideal_lowpass_filter(D0=10, image_size=110):
   filter_function = frequency_bin_index(img_size=image_size)<=D0
   return filter_function




rgb_image = plt.imread('images/farjana.jpeg')
gray_image = make_gray_image(rgb_image)
noisy_image = add_gaussian_noise(gray_image)
image_size = noisy_image.shape[0]

dft_of_image = np.fft.fft2(noisy_image)
dft_of_image = np.fft.fftshift(dft_of_image)
magnitude_of_dft_spectrum = np.abs(dft_of_image)

d1, d2, d3 = 10, 70, 170
ideal_filter_function1 = ideal_lowpass_filter(D0=d1, image_size=image_size)
ideal_filter_function2 = ideal_lowpass_filter(D0=d2, image_size=image_size)
ideal_filter_function3 = ideal_lowpass_filter(D0=d3, image_size=image_size)
ideal_filtered_image1 = dft_of_image * ideal_filter_function1
ideal_filtered_image2 = dft_of_image * ideal_filter_function2
ideal_filtered_image3 = dft_of_image * ideal_filter_function3

magnitude_of_ideal_filtered_1_dft_spectrum = np.abs(ideal_filtered_image1)
magnitude_of_ideal_filtered_2_dft_spectrum = np.abs(ideal_filtered_image2)
magnitude_of_ideal_filtered_3_dft_spectrum = np.abs(ideal_filtered_image3)

dft_of_ideal_filtered_1_image = np.fft.ifftshift(ideal_filtered_image1)
dft_of_ideal_filtered_2_image = np.fft.ifftshift(ideal_filtered_image2)
dft_of_ideal_filtered_3_image = np.fft.ifftshift(ideal_filtered_image3)
ideal_filtered_1_image = np.fft.ifft2(dft_of_ideal_filtered_1_image).real.astype(np.uint8)
ideal_filtered_2_image = np.fft.ifft2(dft_of_ideal_filtered_2_image).real.astype(np.uint8)
ideal_filtered_3_image = np.fft.ifft2(dft_of_ideal_filtered_3_image).real.astype(np.uint8)



plt.figure(figsize=(10, 8))
plt.subplot(5,4,1)
plt.imshow(gray_image, cmap='gray', vmin=0, vmax=255)
plt.title('Original(Spatial domain)')
plt.subplot(5,4,5)
plt.imshow(noisy_image, cmap='gray', vmin=0, vmax=255)
plt.title('Noisy(Spatial domain)')
plt.subplot(5,4,6)
plt.imshow(np.log(1 + magnitude_of_dft_spectrum), cmap='viridis')
plt.title('Noisy(Frequency domain)')
plt.subplot(5,4,7)
plt.plot(np.log(1 + magnitude_of_dft_spectrum))
plt.title('2D Magnitude Spectrum')
plt.subplot(5,4,12)
plt.plot(ideal_filter_function1)
plt.title(f'ideal lowpass D0: {d1}')
plt.subplot(5,4,11)
plt.imshow(ideal_filter_function1, cmap='viridis')
plt.title(f'ideal lowpass D0: {d1}')
plt.subplot(5,4,10)
plt.imshow(np.log(1 + magnitude_of_ideal_filtered_1_dft_spectrum), cmap='viridis')
plt.title('filtered(Frequency domain)')
plt.subplot(5,4,9)
plt.imshow(ideal_filtered_1_image, cmap='gray', vmin=0, vmax=255)
plt.title('filtered(Spatial domain)')
plt.subplot(5,4,16)
plt.plot(ideal_filter_function2)
plt.title(f'ideal lowpass D0: {d2}')
plt.subplot(5,4,15)
plt.imshow(ideal_filter_function2, cmap='viridis')
plt.title(f'ideal lowpass D0: {d2}')
plt.subplot(5,4,14)
plt.imshow(np.log(1 + magnitude_of_ideal_filtered_2_dft_spectrum), cmap='viridis')
plt.title('filtered(Frequency domain)')
plt.subplot(5,4,13)
plt.imshow(ideal_filtered_2_image, cmap='gray', vmin=0, vmax=255)
plt.title('filtered(Spatial domain)')
plt.subplot(5,4,20)
plt.plot(ideal_filter_function3)
plt.title(f'ideal lowpass D0: {d3}')
plt.subplot(5,4,19)
plt.imshow(ideal_filter_function3, cmap='viridis')
plt.title(f'ideal lowpass D0: {d3}')
plt.subplot(5,4,18)
plt.imshow(np.log(1 + magnitude_of_ideal_filtered_3_dft_spectrum), cmap='viridis')
plt.title('filtered(Frequency domain)')
plt.subplot(5,4,17)
plt.imshow(ideal_filtered_3_image, cmap='gray', vmin=0, vmax=255)
plt.title('filtered(Spatial domain)')

plt.tight_layout()
plt.show()

