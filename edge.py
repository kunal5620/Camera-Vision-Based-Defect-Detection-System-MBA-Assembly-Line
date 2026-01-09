import cv2
import matplotlib.pyplot as plt

# input image
image_path = 'D:/ANAND Project/Haldex MBA Project/images for alignment/IMG20240925151514_BURST001_COVER.jpg'  # image path
img = cv2.imread(image_path)

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply the Canny edge detector
edges = cv2.Canny(gray_img, threshold1=100, threshold2=200)

# Save or display the edge-detected image
edge_image_path = 'edge_image_align.jpg'
cv2.imwrite(edge_image_path, edges)

# Show original and edge images using matplotlib
plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(122), plt.imshow(edges, cmap='gray'), plt.title('Edge Image')
plt.show()
