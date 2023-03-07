import numpy as np
from os.path import isfile, join


def fullNucImage16(nucPath):
    scalingFactor = 10000
    for i in range(4):
        #build row block
        nuc1 =np.load(join(nucPath, "Cropped_IMG-" + str(4*i +1) + '_seg.npy'), allow_pickle=True).item()['masks']
        nuc2 =np.load(join(nucPath, "Cropped_IMG-" + str(4*i +2) + '_seg.npy'), allow_pickle=True).item()['masks']
        nuc3 =np.load(join(nucPath, "Cropped_IMG-" + str(4*i +3) + '_seg.npy'), allow_pickle=True).item()['masks']
        nuc4 =np.load(join(nucPath, "Cropped_IMG-" + str(4*i +4) + '_seg.npy'), allow_pickle=True).item()['masks']
        nuc1 = nuc1 + scalingFactor *(4*i +1)
        nuc1[nuc1 == (scalingFactor *(4*i +1))] = 0
        nuc2 = nuc2 + scalingFactor *(4*i +2)
        nuc2[nuc2 == (scalingFactor *(4*i +2))] = 0
        nuc3 = nuc3 + scalingFactor *(4*i +3)
        nuc3[nuc3 == (scalingFactor *(4*i +3))] = 0
        nuc4 = nuc4 + scalingFactor *(4*i +4)
        nuc4[nuc4 == (scalingFactor *(4*i +4))] = 0

        row = np.concatenate((nuc1, nuc2, nuc3, nuc4), axis = 1)
        if i == 0:
            nucImage = row
        else:
            nucImage = np.concatenate((nucImage, row), axis = 0)
    return nucImage


def middleNucImage(nucPath):
    scalingFactor = 10000
    for i in range(4):
        nuc1 =np.load(join(nucPath, "Cropped_IMG-" + str(4*i +1) + '_seg.npy'), allow_pickle=True).item()['masks']
        nuc2 =np.load(join(nucPath, "Cropped_IMG-" + str(4*i +2) + '_seg.npy'), allow_pickle=True).item()['masks']
        nuc3 =np.load(join(nucPath, "Cropped_IMG-" + str(4*i +3) + '_seg.npy'), allow_pickle=True).item()['masks']
        nuc4 =np.load(join(nucPath, "Cropped_IMG-" + str(4*i +4) + '_seg.npy'), allow_pickle=True).item()['masks']
        nuc1 = np.zeros(nuc1.shape)
        nuc4 = np.zeros(nuc4.shape)
        #build row block
        if (i == 0) or (i == 3):
            nuc2 = np.zeros(nuc2.shape)
            nuc3 = np.zeros(nuc3.shape)
        else:
            nuc2 = nuc2 + scalingFactor *(4*i +2)
            nuc2[nuc2 == (scalingFactor *(4*i +2))] = 0
            nuc3 = nuc3 + scalingFactor *(4*i +3)
            nuc3[nuc3 == (scalingFactor *(4*i +3))] = 0


        row = np.concatenate((nuc1, nuc2, nuc3, nuc4), axis = 1)
        if i == 0:
            nucImage = row
        else:
            nucImage = np.concatenate((nucImage, row), axis = 0)
    return nucImage


def outsideNucImage(nucPath):
    scalingFactor = 10000
    for i in range(4):
        nuc1 =np.load(join(nucPath, "Cropped_IMG-" + str(4*i +1) + '_seg.npy'), allow_pickle=True).item()['masks']
        nuc2 =np.load(join(nucPath, "Cropped_IMG-" + str(4*i +2) + '_seg.npy'), allow_pickle=True).item()['masks']
        nuc3 =np.load(join(nucPath, "Cropped_IMG-" + str(4*i +3) + '_seg.npy'), allow_pickle=True).item()['masks']
        nuc4 =np.load(join(nucPath, "Cropped_IMG-" + str(4*i +4) + '_seg.npy'), allow_pickle=True).item()['masks']
        nuc2 = np.zeros(nuc2.shape)
        nuc3 = np.zeros(nuc3.shape)
        #build row block
        nuc1 = nuc1 + scalingFactor *(4*i +1)
        nuc1[nuc1 == (scalingFactor *(4*i +1))] = 0
        nuc4 = nuc4 + scalingFactor *(4*i +4)
        nuc4[nuc4 == (scalingFactor *(4*i +4))] = 0
        row = np.concatenate((nuc1, nuc2, nuc3, nuc4), axis = 1)
        if i == 0:
            nucImage = row
        else:
            nucImage = np.concatenate((nucImage, row), axis = 0)
    return nucImage


def fullNucImageStitched(nucPath, imagePath):
    scalingFactor = 10000
    im = Image.open(imagePath)
    cropping_parameter_width = int(im.size[0]/550)
    cropping_parameter_height = int(im.size[1]/550)
    imgwidth, imgheight = im.size
    crop_width = imgwidth/cropping_parameter_width
    crop_height = imgheight/cropping_parameter_height
    currentHeight = 0
    for i in range(cropping_parameter_height):
        currentWidth = 0
        for j in range(cropping_parameter_width):
            imageNumber = i*(cropping_parameter_width) +j+1
            print(imageNumber)
            if os.path.isfile(join(nucPath, "Cropped_IMG-" + str(imageNumber) + '_seg.npy')):
                nuc = np.load(join(nucPath, "Cropped_IMG-" + str(imageNumber) + '_seg.npy'), allow_pickle=True).item()['masks']
                nuc = nuc + scalingFactor *((cropping_parameter_width*i) + j + 1)
                currentWidth += len(nuc[0])
            else:
                if i == 0:
                    if j == 0:
                        nuc = np.zeros((int(round(crop_height)), int(round(crop_width))))
                        currentWidth += int(round(crop_width))
                    else:
                        nuc = np.zeros((int(round(crop_height)), int(round((j+1)*crop_width - currentWidth))))
                        currentWidth += int(round((j+1)*crop_width - currentWidth))
                else:
                    if j == 0:
                        nuc = np.zeros((int(round((i+1)*crop_height - currentHeight)), int(round(crop_width))))
                        currentWidth += int(round(crop_width))
                    else:
                        nuc = np.zeros((int(round((i+1)*crop_height - currentHeight)), int(round((j+1)*crop_width - currentWidth))))
                        currentWidth += int(round((j+1)*crop_width - currentWidth))

            if j == 0:
                row = nuc
            else:
                row = np.concatenate((row, nuc), axis= 1)

        currentHeight += len(row)

        if i == 0:
            nucImage = row
        else:
            nucImage = np.concatenate((nucImage, row), axis = 0)

    return nucImage