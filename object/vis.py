import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
import matplotlib.cm as cmx
import matplotlib.colors as mcolors


def show_img(im, fig_size=None, ax=None):
    """
    Display the image only
    :param im: image array
    :param fig_size:
    :param ax:
    :return:
    """
    if not ax:
        fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(im)
    # ax.set_xticks(np.linspace(0, 224, 8))
    # ax.set_yticks(np.linspace(0, 224, 8))
    # ax.grid()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    return ax


def draw_outline(o, lw):
    """
    Draws something with outline
    :param o: Outlined object
    :param lw: Line width
    :return:
    """
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])


def draw_rect(ax, box, color='white', line_width=2):
    """
    Draws an outlined rectangle
    :param ax:
    :param box: Bounding box
    :param color:
    :return:
    """
    patch = ax.add_patch(patches.Rectangle(box[:2], *box[-2:], fill=False, edgecolor=color, lw=line_width))
    draw_outline(patch, 4)


def draw_text(ax, xy, txt, sz=14, color='white'):
    text = ax.text(*xy, txt, verticalalignment='top', color=color, fontsize=sz, weight='bold')
    draw_outline(text, 1)


def show_ground_truth(ax, item):
    """

    :param ax:
    :param item:
    :return:
    """
    image = item['image']
    scene = item['scene']
    ax = show_img(image, ax=ax)

    for image_object in scene.objects:
        category_name = image_object.category_name
        bounding_box = image_object.bounding_box
        draw_rect(ax, bounding_box, color='white')
        # Draw the text in the top-left corner of the box
        draw_text(ax, bounding_box[:2], category_name, color='white')

    ax.axis('off')


def show_grid(sample, file_name='plot.png'):
    """

    :param sample: List of ImageItem objects
    :return:
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for i, ax in enumerate(axes.flat):
        show_ground_truth(ax, sample[i])
    plt.tight_layout()
    plt.savefig(file_name)
