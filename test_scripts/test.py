import sys
from pathlib import Path

import cv2
import numpy as np
#read an image with opencv and plot a list of points onto it and then save it

def plot_points_on_image(image, points, output_path):
    """
    Plot a list of points on an image
    :param image:       Image to plot points on
    :param points:      List of points to plot
    :param output_path: Path to save plotted image
    :return:            None
    """
    image_with_points = image.copy()

    # Plot points on image
    for point in points:
        cv2.circle(image_with_points, (point[0], point[1]), 5, (0, 0, 255), -1)

    cv2.imwrite(str(output_path), image_with_points)


if len(sys.argv) < 2:
    raise SystemExit("Usage: python test_scripts/test.py <image_filename>")

image_name = sys.argv[1]
image_path = Path(__file__).resolve().parent.parent / 'binary_images' / image_name
output_path = Path(__file__).resolve().parent / f"annotated_{image_name}"

img = cv2.imread(str(image_path))
if img is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")
points = np.array([[100, 450], [300, 450], [800, 400], [750, 650], [500, 300], [500, 650], [0, 0], [500, 500]])
labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
plot_points_on_image(img, points, output_path)
print(f"Saved annotated image to: {output_path}")
