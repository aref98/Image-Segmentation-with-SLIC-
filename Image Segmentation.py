import os
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import os

def segment_image(image_path, n_segments=100, compactness=10):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_float = img_as_float(image_rgb)

    # Perform SLIC segmentation
    segments = slic(image_float, n_segments=n_segments, compactness=compactness, start_label=1)

    # Create an output image to visualize the segmentation
    segmented_image = label2rgb(segments, image_rgb, kind='avg')

    return segmented_image, segments

if __name__ == "__main__":

    current_directory = os.getcwd()
    print(f"Current working directory: {current_directory}")
    # Define the input and output paths
    input_image_path = "Images/content_image.png"
    output_image_path = "./Images/segmented_image.png"
    output_segments_path = "./Images/segments.npy"

    # Segment the image
    segmented_image, segments = segment_image(input_image_path, n_segments=200, compactness=10)

    # Save the segmented image
    segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path, segmented_image_bgr)

    # Save the segments
    np.save(output_segments_path, segments)

    # Display the segmented image
    plt.imshow(segmented_image)
    plt.title('SLIC Segmentation')
    plt.axis('off')
    plt.show()
