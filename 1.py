import cv2
import numpy as np
from matplotlib import pyplot as plt

# Bước 1: Đọc ảnh đầu vào
image = cv2.imread('hq720.jpg', cv2.IMREAD_GRAYSCALE)

# Bước 2: Sử dụng toán tử Sobel để tính gradient
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient theo hướng X
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient theo hướng Y

# Tính độ lớn của gradient tổng hợp
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Bước 3: Áp dụng toán tử Laplacian of Gaussian (LoG)
log_kernel = np.array([[0, 0, -1, 0, 0],
                       [0, -1, -2, -1, 0],
                       [-1, -2, 16, -2, -1],
                       [0, -1, -2, -1, 0],
                       [0, 0, -1, 0, 0]], dtype=np.float32)

log_image = cv2.filter2D(image, -1, log_kernel)

# Bước 4: Hiển thị ảnh
plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 3, 2), plt.imshow(sobel_combined, cmap='gray'), plt.title('Sobel Edge Detection')
plt.subplot(1, 3, 3), plt.imshow(log_image, cmap='gray'), plt.title('Laplacian of Gaussian')
plt.show()