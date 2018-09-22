# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2

import pandas as pd

from scipy.spatial.distance import cdist

class ColorLabeler:
    def __init__(self):
        # initialize the colors dictionary, containing the color
        # name as the key and the RGB tuple as the value
        color = pd.read_csv('Generic_color.csv')
        color['Code'] = color.index.values

        self.colorFrame = color.copy()

        color['color_value'] = color[['R', 'G', 'B']].apply(tuple, axis=1)
        color_label = {}
        for index, row in color.iterrows():
            color_label[row['Code']] = row['color_value']
        colors = OrderedDict(color_label)
        #		print(colors)

        # allocate memory for the L*a*b* image, then initialize
        # the color names list
        self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
        self.colorNames = []

        # loop over the colors dictionary
        for (i, (name, rgb)) in enumerate(colors.items()):
            # update the L*a*b* array and the color names list
            self.lab[i] = rgb
            self.colorNames.append(name)

        # convert the L*a*b* array from the RGB color space
        # to L*a*b*
        # print ('name', self.lab)

        #self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

    def label(self, mean):
        # construct a mask for the contour, then compute the
        # average L*a*b* value for the masked region
        #		mask = np.zeros(image.shape[:2], dtype="uint8")
        #		cv2.drawContours(mask, [c], -1, 255, -1)
        #		mask = cv2.erode(mask, None, iterations=2)
        #		mean = cv2.mean(image, mask=mask)[:3]

        # initialize the minimum distance found thus far
        minDist = (np.inf, None)
        #		print('last', self.lab[5])

#        mean_init = np.zeros((1, 1, 3), dtype="uint8")
#        mean_init[0] = mean
#        mean_lab = cv2.cvtColor(mean_init, cv2.COLOR_RGB2LAB)
        for (i, row) in enumerate(self.lab):
            # compute the distance between the current L*a*b*
            # color value and the mean of the image
            print(row[0], mean)

            d = dist.euclidean(row[0], mean)
            #d = cdist(row[0], mean, 'minkowski', p=2.)

            #d = cdist(row, mean)

            # if the distance is smaller than the current distance,
            # then update the bookkeeping variable
            if d < minDist[0]:
                print ('row', row)
                minDist = (d, i)

        val = self.colorFrame.iloc[minDist[1]]
        # return the name of the color with the smallest distance
        return val
