import glob
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from utils import get_data


def viz(ground_truth):
    """
    create a grid visualization of images with color coded bboxes
    args:
    - ground_truth [list[dict]]: ground truth data
    """
    paths = glob.glob('data/images/*')

    # mapping to access data faster
    gtdic = {}
    for gt in ground_truth:
        gtdic[gt['filename']] = gt

    # color mapping of classes
    colormap = {1: [1, 0, 0], 2: [0, 1, 0], 4: [0, 0, 1]}

    f, ax = plt.subplots(4, 5, figsize=(20, 10))
    for i in range(20):
        x = i // 5  # Corrected indexing for rows
        y = i % 5   # Corrected indexing for columns

        filename = os.path.basename(paths[i])
        img = Image.open(paths[i])
        ax[x, y].imshow(img)

        bboxes = gtdic[filename]['boxes']
        classes = gtdic[filename]['classes']
        for cl, bb in zip(classes, bboxes):
            y1, x1, y2, x2 = bb
            rec = Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor='none',
                            edgecolor=colormap[cl], linewidth=2)
            ax[x, y].add_patch(rec)
        ax[x, y].axis('off')

    plt.tight_layout()

    # create output folder if it doesn't exist
    output_folder = 'Output_Visualization'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # save the plot
    output_filepath = os.path.join(output_folder, 'visualization_result.png')
    plt.savefig(output_filepath)
    print(f'Visualization saved at {output_filepath}')

    #plt.show()


if __name__ == "__main__":
    ground_truth, _ = get_data()
    viz(ground_truth)