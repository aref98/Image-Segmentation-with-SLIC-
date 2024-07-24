import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float32
import matplotlib.pyplot as plt

def apply_clahe(image_channel, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the given image channel.

    Parameters:
    - image_channel: Single channel of the image.
    - clip_limit: Threshold for contrast limiting.
    - tile_grid_size: Size of grid for histogram equalization.

    Returns:
    - clahe_channel: Enhanced image channel after applying CLAHE.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image_channel)

def enhance_segments(image, segments):
    """
    Enhance the colors of each segment to make them more distinct by adjusting LAB values
    and applying CLAHE to the luminance channel.

    Parameters:
    - image: Original image.
    - segments: Array with segment labels.

    Returns:
    - enhanced_image: Image with enhanced segments.
    """
    # Convert to LAB color space
    image_lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    enhanced_image_lab = np.copy(image_lab)

    # Apply CLAHE to the L (luminance) channel
    enhanced_image_lab[:, :, 0] = apply_clahe(image_lab[:, :, 0])

    for segment_value in np.unique(segments):
        mask = segments == segment_value
        if np.sum(mask) == 0:
            continue

        # Apply the mask to each channel separately
        l = image_lab[:, :, 0][mask]
        a = image_lab[:, :, 1][mask]
        b = image_lab[:, :, 2][mask]

        # Increase a and b values to make colors more vibrant
        a = np.clip(a + 10, 0, 255)
        b = np.clip(b + 10, 0, 255)

        # Put the enhanced values back
        enhanced_image_lab[:, :, 1][mask] = a
        enhanced_image_lab[:, :, 2][mask] = b

    # Convert back to RGB
    enhanced_image_rgb = cv2.cvtColor(enhanced_image_lab, cv2.COLOR_LAB2RGB) / 255.0
    
    return enhanced_image_rgb

if __name__ == "__main__":
    # Define the input image and segments paths
    input_image_path = "Images/content_image.png"
    segments_path = "Images/segments.npy"

    # Load the original image and segmented image
    image = cv2.imread(input_image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_float32 = img_as_float32(image_rgb)

    # Load the saved segments array
    segments = np.load(segments_path)

    # Enhance the segments using LAB adjustment with CLAHE
    enhanced_image = enhance_segments(image_float32, segments)

    # Convert the enhanced image to uint8 for saving
    enhanced_image_uint8 = (enhanced_image * 255).astype(np.uint8)

    # Save and display the enhanced image
    enhanced_image_bgr = cv2.cvtColor(enhanced_image_uint8, cv2.COLOR_RGB2BGR)
    output_image_path = "Images/enhanced_image.png"
    cv2.imwrite(output_image_path, enhanced_image_bgr)

    # Display the enhanced image
    plt.imshow(enhanced_image)
    plt.title('Enhanced Segments')
    plt.axis('off')
    plt.show()
