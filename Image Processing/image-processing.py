
import numpy as np
import cv2

kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernel_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

def apply_gamma_correction(image, gamma):
    lookup_table = np.array([(i / 255.0) ** gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    corrected_image = cv2.LUT(image, lookup_table)
    return corrected_image

def detect_edges_prewitt(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges_x = cv2.filter2D(gray_image, -1, kernel_x)
    edges_y = cv2.filter2D(gray_image, -1, kernel_y)
    edges_image = cv2.addWeighted(edges_x, 0.5, edges_y, 0.5, 0)
    
    edges_image_color = cv2.cvtColor(edges_image, cv2.COLOR_GRAY2BGR)
    return edges_image_color

def apply_dct(image):
    channels = [cv2.dct(np.float32(image[:, :, c])) for c in range(3)]
    normalized_channels = [cv2.normalize(channel, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U) for channel in channels]
    return cv2.merge(normalized_channels)

def main():
    image = cv2.imread('pengu.jpg')
    
    gamma_corrected_image = apply_gamma_correction(image.copy(), 0.5)
    cv2.imwrite('gamma_corrected.jpg', gamma_corrected_image)
    
    edges_detected_image = detect_edges_prewitt(image.copy())
    cv2.imwrite('edges_detected.jpg', edges_detected_image)
    
    dct_transformed_image = apply_dct(image.copy())
    cv2.imwrite('dct_transformed.jpg', dct_transformed_image)

if __name__ == "__main__":
    main()
