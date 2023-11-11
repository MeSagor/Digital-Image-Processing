import numpy as np
import matplotlib.pyplot as plt

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

def ideal_highpass_filter(D0=10, image_size=110):
   D = frequency_bin_index(img_size=image_size)
   filter_function = D > D0
   return filter_function

def gaussian_highpass_filter(D0=10, image_size=110):
   D = frequency_bin_index(img_size=image_size)
   filter_function = 1 - np.exp(-(D**2) / (2*D0**2))
   return filter_function



rgb_image = plt.imread('images/star.jpg')
gray_image = make_gray_image(rgb_image)
noisy_image = add_gaussian_noise(gray_image)
image_size = noisy_image.shape[0]


dft_original_image = np.fft.fftshift(np.fft.fft2(gray_image))
dft_noisy_image = np.fft.fftshift(np.fft.fft2(noisy_image))

D = 20
filter_original_image_IHPF = dft_original_image * ideal_highpass_filter(D, image_size)
filter_original_image_GHPF = dft_original_image * gaussian_highpass_filter(D, image_size)

filter_noisy_image_IHPF = dft_noisy_image * ideal_highpass_filter(D, image_size)
filter_noisy_image_GHPF = dft_noisy_image * gaussian_highpass_filter(D, image_size)

filter_original_image_IHPF = np.fft.ifftshift(filter_original_image_IHPF)
filter_original_image_GHPF = np.fft.ifftshift(filter_original_image_GHPF)
filter_noisy_image_IHPF = np.fft.ifftshift(filter_noisy_image_IHPF)
filter_noisy_image_GHPF = np.fft.ifftshift(filter_noisy_image_GHPF)

filter_original_image_IHPF = np.fft.ifft2(filter_original_image_IHPF).real.astype(np.uint8)
filter_original_image_GHPF = np.fft.ifft2(filter_original_image_GHPF).real.astype(np.uint8)
filter_noisy_image_IHPF = np.fft.ifft2(filter_noisy_image_IHPF).real.astype(np.uint8)
filter_noisy_image_GHPF = np.fft.ifft2(filter_noisy_image_GHPF).real.astype(np.uint8)


plt.figure(figsize=(9, 7))

plt.subplot(231)
plt.imshow(gray_image, cmap='gray')
plt.title('Original')
plt.subplot(232)
plt.imshow(filter_original_image_IHPF, cmap='gray')
plt.title('Filtered by IHPF')
plt.subplot(233)
plt.imshow(filter_original_image_GHPF, cmap='gray')
plt.title('Filtered by GHPF')
plt.subplot(234)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy')
plt.subplot(235)
plt.imshow(filter_noisy_image_IHPF, cmap='gray')
plt.title('Filtered by IHPF')
plt.subplot(236)
plt.imshow(filter_noisy_image_GHPF, cmap='gray')
plt.title('Filtered by GHPF')

plt.tight_layout()
plt.show()




