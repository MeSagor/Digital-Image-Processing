import numpy as np
import matplotlib.pyplot as plt


def make_gray_image(image):
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    return gray_image


def brightness_enhancement(image, min_intencity, max_intencity, enhancement_factor):
    enhanced_image = np.copy(image)

    for y in range(enhanced_image.shape[0]):
        for x in range(enhanced_image.shape[1]):
            gray_value = enhanced_image[y, x]
            if gray_value >= min_intencity and gray_value <= max_intencity:
                new_gray_value = gray_value + enhancement_factor
                if new_gray_value > 255:
                    new_gray_value = 255
                elif new_gray_value < 0:
                    new_gray_value = 0
                enhanced_image[y, x] = new_gray_value

    return enhanced_image


rgb_image = plt.imread('images/skull.jpg')
gray_image = make_gray_image(rgb_image)

low, high, factor = 150, 205, 50
enhanced_image = brightness_enhancement(gray_image, low, high, factor)


plt.figure(figsize=(8, 7))
plt.subplot(121)
plt.imshow(gray_image, cmap='gray', vmin=0, vmax=255)
plt.title('Original')
plt.subplot(122)
plt.imshow(enhanced_image, cmap='gray', vmin=0, vmax=255)
plt.title(f'Enhanced between[{low}-{high}] by {factor}')

plt.tight_layout()
plt.show()
