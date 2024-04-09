import sys
import os
from PIL import Image
import numpy as np

def process_images(image1_path, image2_path, image3_path, image4_path, N):
    # Load images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    image3 = Image.open(image3_path)
    image4 = Image.open(image4_path)

    # Resize image1
    image1_resized = image1.resize((9 * N, 9 * N), Image.NEAREST)

    # Resize image2
    height = int(N)
    width = int(height / image2.height * image2.width)
    image2_resized = image2.resize((width, height), Image.NEAREST)

    # Create a new blank image
    final_image = Image.new('RGB', (9 * N, 9 * N))

    final_image.paste(image1_resized, (0, 0))

    # Paste image2 onto image1 (centered at the top)
    x_offset = int((9 * N - width) / 2)
    final_image.paste(image2_resized, (x_offset, 0))

    # Paste image3 onto image1
    image3_resized = image3.resize((N, N), Image.NEAREST)
    final_image.paste(image3_resized, (3 * N, 5 * N))

    # Paste image4 onto image1
    image4_resized = image4.resize((N, N), Image.NEAREST)
    final_image.paste(image4_resized, (5 * N, 5 * N))

    # Calculate crop size
    crop_size = int(0.25 * N)

    # Crop the final image
    final_image = final_image.crop((crop_size, crop_size, 9 * N - crop_size, 9 * N - crop_size))

    # Get the base names of the images
    image1_base = os.path.basename(image1_path)
    image3_base = os.path.basename(image3_path)
    image4_base = os.path.basename(image4_path)

    # Save the final image
    output_path = "{}_{}_{}.png".format(image1_base[:-4], image3_base[:-4], image4_base[:-4])
    final_image.save(output_path, "PNG")

    print("Image saved as", output_path)

# Check if the correct number of command line arguments is provided
if len(sys.argv) != 5:
    print("Usage: python tag_combine.py tag25h9/tag25_09_00000.png logo.png tag16h5/tag16_05_00000.png tag16h5/tag16_05_00001.png")
else:
    # Get command line arguments
    image1_path = sys.argv[1]
    image2_path = sys.argv[2]
    image3_path = sys.argv[3]
    image4_path = sys.argv[4]
    N = 1000

    # Process images
    process_images(image1_path, image2_path, image3_path, image4_path, N)
