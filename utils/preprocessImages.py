import numpy as np
import cv2
import skimage.io
from os.path import isfile, join
from os import listdir
from PIL import Image
from skimage import img_as_ubyte, exposure
import fnmatch

def mergeImageChannelsNormalize(pathChannel1, pathChannel2, pathChannel3 = None, save = True, savePath, filename):
    channel1Image = skimage.io.imread(pathChannel1)
    channel1Image_normalized = img_as_ubyte(exposure.rescale_intensity(channel1Image))
    channel2Image = skimage.io.imread(pathChannel2)
    channel2Image_normalized = img_as_ubyte(exposure.rescale_intensity(channel2Image))
    if pathChannel3 not None:
        channel3Image = skimage.io.imread(pathChannel3)
        channel3Image_normalized = img_as_ubyte(exposure.rescale_intensity(channel3Image))
    else:
        zeros = np.zeros(nuclei.shape, dtype="uint8")
        channel3Image_normalized = zeros
    mergedImage= cv2.merge([channel1Image_normalized, channel2Image_normalized, channel3Image_normalized])
    im = Image.fromarray(mergedImage)
    if save = True:
        im.save(join(savePath,filename))
    return im

def cropImage(filename, path_to_save, cropping_parameter):
    im = Image.open(filename)
    k=1
    imgwidth, imgheight = im.size
    crop_width = imgwidth/cropping_parameter
    crop_height = imgheight/cropping_parameter
    for i in range(cropping_parameter):
        for j in range(cropping_parameter):
            box = (j*crop_width,i*crop_height,(j+1)*crop_width, (i+1)*crop_height)
            crop = im.crop(box)
            crop.save(os.path.join(path_to_save,"Cropped_IMG-%s.tif" % k))
            k +=1



def mergingBlueNuclei_aboveInt(bluePath, nucleiPath, savePath):
    blueFiles = [f for f in listdir(bluePath) if isfile(join(bluePath,f))]
    nucleiFiles = [f for f in listdir(nucleiPath) if isfile(join(nucleiPath,f))]
    for i in range(len(blueFiles)):
        patternBlue = "*-" + str(i+1) + '.tif'
        patternNuclei = "*-" + str(i+1) + '_seg.npy'
        blueFile = fnmatch.filter(blueFiles, patternBlue)
        nucleiFile = fnmatch.filter(nucleiFiles, patternNuclei)
        blueImage = skimage.io.imread(join(bluePath,blueFile[0]))
        nuclei = np.load(join(nucleiPath,nucleiFile[0]), allow_pickle=True).item()
        nucleiMasks = nuclei['masks']
        nucleiMaskBlueInt = np.where(nucleiMasks > 0, blueImage, 0 )
        nucleiMaskOnlyBlue = np.where(nucleiMaskBlueInt > 1500, nucleiMaskBlueInt, 0)
        zeros = np.zeros(blueImage.shape, dtype="uint8")
        dapi8bitNormalized= img_as_ubyte(exposure.rescale_intensity(nucleiMaskOnlyBlue))
        blueChannel8bitNormalized = img_as_ubyte(exposure.rescale_intensity(blueImage))
        merged8bitNormalized = cv2.merge([zeros, dapi8bitNormalized, blueChannel8bitNormalized])
        im = Image.fromarray(merged8bitNormalized)
        filename = 'mergedBlueNuclei' + str(i+1) + '.tif'
        im.save(join(savePath,filename))
