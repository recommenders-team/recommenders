# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Author: Graham.Williams@togaware.com

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import toolz
from PIL import Image, ImageOps
import re
from io import BytesIO

def read_image_from(fname):
    return toolz.pipe(
        fname,
        lambda x: open(x, 'rb'),
        lambda x: x.read(),
        BytesIO)

def to_rgb(img_bytes):
    return Image.open(img_bytes).convert('RGB')

def to_img(img_fname):
    return toolz.pipe(img_fname,
                      read_image_from,
                      to_rgb)

def _plot_image(ax, img):
    ax.imshow(to_img(img))
    ax.tick_params(axis='both',
                   which='both',
                   bottom=False,
                   top=False,
                   left=False,
                   right=False,
                   labelleft=False,
                   labelbottom=False)
    return ax

def plot_recommendations():

    images = ["m0.jpg", "m1.jpg", "m2.jpg", "m3.jpg", "m4.jpg",
              "p0.jpg", "p1.jpg", "p2.jpg", "p3.jpg", "p4.jpg"]

    gs = gridspec.GridSpec(2, 5)
    fig = plt.figure(figsize=(6, 3))
    gs.update(hspace=0.1, wspace=0.001)

    for i in range(10):
        gg2 = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=gs[i])
        ax  = fig.add_subplot(gg2[0:3, :])
        _plot_image(ax, images[i])

    # fig = plt.figure()
    # fig.savefig("sample.png")
    plt.show()

