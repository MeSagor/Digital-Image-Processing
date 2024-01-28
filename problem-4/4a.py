import numpy as np
import matplotlib.pyplot as plt

def make_gray_image(image):
    gray_image = image
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = np.dot(image[..., :3], [0.30, 0.59, 0.11]).astype(np.uint8)
    return gray_image

def custom_gray_image_generator():
    custom_image = np.empty((81, 81), dtype=np.uint8)
    val = 0
    for i in range(81):
        for j in range(81):
            custom_image[i, j] = 0

            if i>30 and j>30 and i<50 and j<50:
                custom_image[i, j] = 255
            # if i==j:
            #     custom_image[i, j] = 255

            # val += 1
    return custom_image

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

def gaussian_lowpass_filter(D0=10, image_size=110):
   filter_function = np.exp(-(frequency_bin_index(img_size=image_size)**2) / (2*D0**2))
   return filter_function

def butterworth_lowpass_filter(D0=10, order=3, image_size=110):
   filter_function = 1/(1 + (frequency_bin_index(img_size=image_size)/D0)**(2*order))
   return filter_function

def calculate_dft(image):
    height, width = image.shape
    dft_image = np.zeros((height, width), dtype=np.complex128)

    for u in range(height):
        for v in range(width):
            print(f'DFT processing -> Row: {u} Col: {v}', end='\r')
            # dft_sum = 0
            # for x in range(height):
            #     for y in range(width):
            #         pixel_value = image[x, y]
            #         dft_sum += pixel_value * np.exp(-2j * np.pi * ((u * x / height) + (v * y / width)))
            # dft_image[u, v] = dft_sum
            x, y = np.meshgrid(np.arange(height), np.arange(width))
            dft_image[u, v] = np.sum(image * np.exp(-2j * np.pi * ((u * x / height) + (v * y / width))))
    return dft_image

def calculate_idft(dft_image):
    height, width = dft_image.shape
    idft_image = np.zeros((height, width), dtype=np.complex128)

    for x in range(height):
        for y in range(width):
            print(f'IDFT Processing -> Row: {x} Col: {y}', end='\r')
            u, v = np.meshgrid(np.arange(height), np.arange(width))
            idft_sum = np.sum(dft_image * np.exp(2j * np.pi * ((u * x / height) + (v * y / width))))
            idft_image[x, y] = idft_sum / (height * width)
    return idft_image.real.astype(np.uint8)

def show_in_3d(magnitude_spectrum):
    height, width = magnitude_spectrum.shape
    u_values = np.fft.fftshift(np.fft.fftfreq(height))
    v_values = np.fft.fftshift(np.fft.fftfreq(width))
    U, V = np.meshgrid(u_values, v_values)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(U, V, np.log(1 + magnitude_spectrum), cmap='viridis')
    ax.set_xlabel('Frequency (u)')
    ax.set_ylabel('Frequency (v)')
    ax.set_zlabel('Magnitude Spectrum (log scale)')
    ax.set_title('3D Magnitude Spectrum')


rgb_image = plt.imread('images/farjana.jpeg')
gray_image = make_gray_image(rgb_image)
gray_image = custom_gray_image_generator()
# gray_image = (transform.resize(gray_image, (110, 110)) * 255).astype(np.uint8)
noisy_image = add_gaussian_noise(gray_image)
image_size = noisy_image.shape[0]

dft_of_image = calculate_dft(noisy_image)
dft_of_image = np.fft.fftshift(dft_of_image)
magnitude_of_dft_spectrum = np.abs(dft_of_image)

gaussian_filter_function = gaussian_lowpass_filter(D0=20, image_size=image_size)
butterworth_filter_function = butterworth_lowpass_filter(D0=20, order=4, image_size=image_size)
gaussian_filtered_image = dft_of_image * gaussian_filter_function
butterworth_filtered_image = dft_of_image * butterworth_filter_function

magnitude_of_gaussian_filtered_dft_spectrum = np.abs(gaussian_filtered_image)
magnitude_of_butterworth_filtered_dft_spectrum = np.abs(butterworth_filtered_image)

dft_of_gaussian_filtered_image = np.fft.ifftshift(gaussian_filtered_image)
dft_of_butterworth_filtered_image = np.fft.ifftshift(butterworth_filtered_image)
gaussian_filtered_image = calculate_idft(dft_of_gaussian_filtered_image)
butterworth_filtered_image = calculate_idft(dft_of_butterworth_filtered_image)



plt.figure(figsize=(12, 7))
plt.subplot(441)
plt.imshow(gray_image, cmap='gray', vmin=0, vmax=255)
plt.title('Original(Spatial domain)')
plt.subplot(445)
plt.imshow(noisy_image, cmap='gray', vmin=0, vmax=255)
plt.title('Noisy(Spatial domain)')
plt.subplot(446)
plt.imshow(np.log(1 + magnitude_of_dft_spectrum), cmap='viridis')
plt.title('Noisy(Frequency domain)')
plt.subplot(447)
plt.plot(np.log(1 + magnitude_of_dft_spectrum))
plt.title('2D Magnitude Spectrum')
plt.subplot(4,4,12)
plt.plot(gaussian_filter_function)
plt.title('2D gaussian filter function')
plt.subplot(4,4,11)
plt.imshow(gaussian_filter_function, cmap='viridis')
plt.title('gaussian filter function')
plt.subplot(4,4,10)
plt.imshow(np.log(1 + magnitude_of_gaussian_filtered_dft_spectrum), cmap='viridis')
plt.title('filtered(Frequency domain)')
plt.subplot(4,4,9)
plt.imshow(gaussian_filtered_image, cmap='gray', vmin=0, vmax=255)
plt.title('filtered(Spatial domain)')
plt.subplot(4,4,16)
plt.plot(butterworth_filter_function)
plt.title('2D butterworth filter function')
plt.subplot(4,4,15)
plt.imshow(butterworth_filter_function, cmap='viridis')
plt.title('butterworth filter function')
plt.subplot(4,4,14)
plt.imshow(np.log(1 + magnitude_of_butterworth_filtered_dft_spectrum), cmap='viridis')
plt.title('filtered(Frequency domain)')
plt.subplot(4,4,13)
plt.imshow(butterworth_filtered_image, cmap='gray', vmin=0, vmax=255)
plt.title('filtered(Spatial domain)')

# show_in_3d(filter_function)

plt.tight_layout()
plt.show()

