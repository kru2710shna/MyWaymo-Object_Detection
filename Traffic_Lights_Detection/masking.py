import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


def create_mask(path, color_threshold):
    """
    create a binary mask of an image using a color threshold
    """
    img = np.array(Image.open(path).convert('RGB'))
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    rt, gt, bt = color_threshold
    mask = (R > rt) & (G > gt) & (B > bt) 
    return img, mask


def mask_and_display(img, mask, output_path):
    """
    display 3 plots and save masked image
    """
    masked_image = img * np.stack([mask]*3, axis=2)

    # Display
    f, ax = plt.subplots(1, 3, figsize=(15, 10))
    ax[0].imshow(img)
    ax[0].set_title("Original")
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("Binary Mask")
    ax[2].imshow(masked_image)
    ax[2].set_title("Masked Image")
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()

    # Save masked image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(masked_image.astype(np.uint8)).save(output_path)
    print(f"Masked image saved to: {output_path}")


if __name__ == '__main__':
    path = 'data/images/segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    color_threshold = [128, 128, 128]

    # Set output file path
    output_file = '/Users/krushna/MyWaymo-Object_Detection/Traffic_Lights_Detection/Output/Output_Masking/masked_image.png'

    img, mask = create_mask(path, color_threshold)
    mask_and_display(img, mask, output_file)
