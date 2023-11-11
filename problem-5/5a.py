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


def do_dilation(image, structuring_element):
    pad_width = structuring_element.shape[0] // 2
    image = np.pad(image, pad_width, mode='constant')

    dilation = np.zeros_like(image)
    for i in range(pad_width, image.shape[0]-pad_width):
        for j in range(pad_width, image.shape[1]-pad_width):
            window = image[i-pad_width : i+pad_width+1, j-pad_width : j+pad_width+1]
            if np.any(np.logical_and(window, structuring_element)):
                dilation[i, j] = 1

    dilation = dilation[pad_width:-pad_width, pad_width:-pad_width]
    return dilation


rgb_image = plt.imread('./images/box.jpg')
gray_image = make_gray_image(rgb_image)
print(np.max(gray_image))

binary_image = (gray_image > 100).astype(np.uint8)

structuring_element = np.array([[0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0],
                                [1, 1, 1, 1, 1],
                                [0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0]], dtype=np.uint8)

# binary_image = np.array([[0, 0, 0, 0, 0],
#                          [0, 1, 1, 1, 0],
#                          [0, 1, 1, 1, 1],
#                          [0, 1, 1, 1, 0],
#                          [0, 0, 1, 0, 0],
#                          [0, 1, 0, 0, 0],
#                          [0, 0, 0, 0, 0]], dtype=np.uint8)

eroded_image = do_erosion(binary_image, structuring_element)
dilated_image = do_dilation(binary_image, structuring_element)

plt.figure(figsize=(8, 6))
plt.subplot(221)
plt.imshow(binary_image, cmap='gray')
plt.title('Original')
plt.subplot(222)
plt.imshow(eroded_image, cmap='gray')
plt.title('Eroded')
plt.subplot(223)
plt.imshow(binary_image, cmap='gray')
plt.title('Original')
plt.subplot(224)
plt.imshow(dilated_image, cmap='gray')
plt.title('Dilated')

plt.tight_layout()
plt.show()
