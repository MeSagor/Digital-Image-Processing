import numpy as np
import matplotlib.pyplot as plt


def make_gray_image(image):
    gray_image = image
    if len(image.shape)==3:
        gray_image = np.mean(image, axis=2)
    return gray_image

def do_edge_detection(image, sobel_x, sobel_y):
    pad_width = sobel_x.shape[0] // 2
    image = np.pad(image, pad_width, mode='constant')

    horizontal_edges = np.zeros_like(image)
    vertical_edges = np.zeros_like(image)
    for i in range(pad_width, image.shape[0]-pad_width):
        for j in range(pad_width, image.shape[1]-pad_width):
            local_region = image[i-pad_width : i+pad_width+1, j-pad_width : j+pad_width+1]
            horizontal_edges[i, j] = np.sum(local_region * sobel_x)
            vertical_edges[i, j] = np.sum(local_region * sobel_y)
            
    edge_magnitude = np.sqrt(horizontal_edges**2 + vertical_edges**2)
    edge_magnitude = edge_magnitude[pad_width:-pad_width, pad_width:-pad_width]
    return edge_magnitude

rgb_image = plt.imread('images/landscape.jpg')
gray_image = make_gray_image(rgb_image)


sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = sobel_x.T

edge_detected_image = do_edge_detection(gray_image, sobel_x, sobel_y)


plt.figure(figsize=(7,6))
plt.subplot(121)
plt.imshow(gray_image, cmap='gray', vmin=0, vmax=255)
plt.title('Original')
plt.subplot(122)
plt.imshow(edge_detected_image, cmap='gray')
plt.title('Detected')

plt.tight_layout()
plt.show()