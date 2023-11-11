import numpy as np
import matplotlib.pyplot as plt


def make_gray_image(image):
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    return gray_image

def do_point_detection(image, kernel):
    pad_width = kernel.shape[0] // 2
    image = np.pad(image, pad_width, mode='constant')

    detected_image = np.zeros_like(image)
    for i in range(pad_width, image.shape[0]-pad_width):
        for j in range(pad_width, image.shape[1]-pad_width):
            local_region = image[i-pad_width : i+pad_width+1, j-pad_width : j+pad_width+1]
            val = np.sum(local_region * kernel)
            if val > 255:
                detected_image[i, j] = 255
            else:
                detected_image[i, j] = 0
    detected_image = detected_image[pad_width:-pad_width, pad_width:-pad_width]
    return detected_image

rgb_image = plt.imread('images/point.jpg')
gray_image = make_gray_image(rgb_image)

kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

point_detected_image = do_point_detection(gray_image, kernel)


plt.figure(figsize=(7,6))
plt.subplot(121)
plt.imshow(gray_image, cmap='gray', vmin=0, vmax=255)
plt.title('Original')
plt.subplot(122)
plt.imshow(point_detected_image, cmap='gray', vmin=0, vmax=255)
plt.title('Detected')

plt.tight_layout()
plt.show()