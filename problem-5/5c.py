import numpy as np
import matplotlib.pyplot as plt

def make_gray_image(image):
    gray_image = image
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = np.dot(image[..., :3], [0.30, 0.59, 0.11]).astype(np.uint8)
    return gray_image


def do_erosion(image, structuring_element):
    pad_width = structuring_element.shape[0] // 2
    image = np.pad(image, pad_width, mode='constant')

    erosion = np.zeros_like(image)
    for i in range(pad_width, image.shape[0]-pad_width):
        for j in range(pad_width, image.shape[1]-pad_width):
            window = image[i-pad_width : i+pad_width+1, j-pad_width : j+pad_width+1]
            if np.array_equal(np.logical_and(window, structuring_element), structuring_element):
                erosion[i, j] = 1
    
    erosion = erosion[pad_width:-pad_width, pad_width:-pad_width]
    return erosion


rgb_image = plt.imread('./images/shape.jpg')
gray_image = make_gray_image(rgb_image)
print(np.max(gray_image))

binary_image = (gray_image > 100).astype(np.uint8)

structuring_element = np.array([[0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0],
                                [1, 1, 1, 1, 1],
                                [0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0]], dtype=np.uint8)

eroded_image = do_erosion(binary_image, structuring_element)
boundary = binary_image - eroded_image

plt.figure(figsize=(9, 6))
plt.subplot(121)
plt.imshow(binary_image, cmap='gray')
plt.title('Original')
plt.subplot(122)
plt.imshow(boundary, cmap='gray')
plt.title('Boundary')

plt.tight_layout()
plt.show()
