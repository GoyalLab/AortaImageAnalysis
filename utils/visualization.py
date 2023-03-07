import colorsys
import numpy as np


def hsv_to_rgb(arr):
    hsv_to_rgb_channels = np.vectorize(colorsys.hsv_to_rgb)
    h, s, v = np.rollaxis(arr, axis=-1)
    r, g, b = hsv_to_rgb_channels(h, s, v)
    rgb = np.stack((r,g,b), axis=-1)
    return rgb

def mask_overlay(img, masks, colors=None):

    img = img.astype(np.float32)

    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    HSV[:,:,2] = np.clip((img / 255. if img.max() > 1 else img) * 1.5, 0, 1)
    hues = np.linspace(0, 1, masks.max()+1)[np.random.permutation(masks.max())]
    for n in range(int(masks.max())):
        ipix = (masks==n+1).nonzero()
        if colors is None:
            HSV[ipix[0],ipix[1],0] = hues[n]
        else:
            HSV[ipix[0],ipix[1],0] = colors[n,0]
        HSV[ipix[0],ipix[1],1] = 1.0
    RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB

def plotMasks(img, masks):
    img = Image.fromarray(img)
    imgGray = img.convert('L')
    imgGrayarray = np.asarray(imgGray)
    rgb = mask_overlay(imgGrayarray, masks, colors=None)
    fig, ax = plt.subplots(figsize=(36,15))
    ax.imshow(rgb)


def plotImage_Numbers(img, df_plot, outlineImage = True, outlines = None, outlineColor = [255,255,255]):
    #overlay the outlines
    if outlineImage == True:
        if outlines is not None:
            fig, ax = plotImage_Outline(outlines, img, outlineColor = outlineColor, plot = False)
    else:
        fig, ax = plt.subplots(figsize=(36,15))
        ax.imshow(img)
    for i in range(len(df_plot)):
        ax.text(df_plot['centroid-1'][i], df_plot['centroid-0'][i], df_plot['label'][i], fontsize=13, color = 'red')
    plt.show()
