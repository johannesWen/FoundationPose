import cv2
import os
import glob

# Path to the folder with images
image_folder = r'D:\Linz Center of Mechatronics GmbH\BA2_OD1 - BA2_KEBA\S12209\work\Object Tracking\Foundationpose\Jetson Versuche\cup01\track_vis'  # Adjust this path as needed

# Output video file name
output_video = 'output.mp4'

# Frames per Second (number of images per second in the video)
fps = 6

# Get list of image files (supports jpg and png)
images = glob.glob(os.path.join(image_folder, '*.jpg')) + glob.glob(os.path.join(image_folder, '*.png'))

if not images:
    print("No images found in the specified folder.")
    exit()

def all_filenames_numeric(image_list):
    """Check if all image filenames (without extension) are numeric."""
    for image in image_list:
        name = os.path.splitext(os.path.basename(image))[0]
        if not name.isdigit():
            return False
    return True

# If filenames are all integers, sort numerically; otherwise, lexicographically.
if all_filenames_numeric(images):
    images = sorted(images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
else:
    images = sorted(images)

# Read the first image to determine the size
frame = cv2.imread(images[0])
height, width, layers = frame.shape

# Create the VideoWriter with the 'mp4v' codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Write each image as a frame to the video
for image in images:
    frame = cv2.imread(image)
    if frame is None:
        print(f"Error reading {image}. Skipping this image.")
        continue
    video.write(frame)

video.release()
print(f"Video has been created and saved as {output_video}.")
