from imageai.Detection import ObjectDetection
from mrcnn import utils

import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
import matplotlib.patches as patches

from colorlabeler import ColorLabeler
from maskrcnn import *

import warnings
warnings.filterwarnings("ignore")

cl = ColorLabeler()


lst_intensities = []



def preprocess(rgb):
    blurred = cv2.GaussianBlur(rgb, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    #   print ('gray_shape', gray.shape)
    binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                       cv2.THRESH_BINARY, 11, 2)
   # binary = binary - binary.max()
    #binary[binary==255] = 1
    return binary


def find_countours(binary_img):
    _, cnts, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnts = [cnts[0]]
    return cnts


def extract_patch_mask(img, array):
    x1, y1, x2, y2 = array[0], array[1], array[2], array[3]
    sub_img = img[y1:y2, x1:x2]
    return sub_img




def run_kmeans(lst_intensities):

    array = np.array(lst_intensities[0])

    kmeans = KMeans(init='k-means++',n_clusters=3,random_state=0)
    kmeans.fit(array)


    #print (gmm.cluster_centers_)

    kmeans_labels = kmeans.predict(array)

    unique_l, counts_l = np.unique(kmeans.labels_, return_counts=True)

    sort_ix = np.argsort(counts_l)

    sort_ix = sort_ix[::-1]


    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # x_from = 0.05
    # #print(kmeans.cluster_centers_[sort_ix])
    # for cluster_center in kmeans.cluster_centers_[sort_ix]:
    #     print('point RGB', int(cluster_center[2]), int(cluster_center[1]), int(cluster_center[0]))
    #     ax.add_patch(patches.Rectangle((x_from, 0.05), 0.29, 0.9, alpha=None,
    #                                    facecolor='#%02x%02x%02x' % (
    #                                    int(cluster_center[2]), int(cluster_center[1]), int(cluster_center[0]))))
    #     x_from = x_from + 0.31
    
    # plt.show()
    return kmeans.cluster_centers_[sort_ix][0]

def make_segmentation_mask(image, mask):
    img = image.copy()
    img[:,:,0] *= mask
    img[:,:,1] *= mask
    img[:,:,2] *= mask
    return img


def find_pixels(rcnn_result, mask):
    list_intensities = []
    mask = mask*1
    pts = np.where(mask == 1)
    list_intensities.append(rcnn_result[pts[0], pts[1]])
    return list_intensities


if __name__ == "__main__":
    
    file_name = '0a2bbd5330a2_08.jpg'
    execution_path = os.getcwd()
    img = cv2.imread(os.path.join(execution_path, file_name))
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=img,
                                                 output_image_path=
                                                 os.path.join(execution_path, "imagenew.jpg"),
                                                 input_type='array')

    temp = 0

    if len(detections) !=0:
        for eachObject in detections:
            if eachObject['name'] == 'car' or eachObject['name'] == 'truck':
                area = (eachObject['box_points'][2]- eachObject['box_points'][0])+ (eachObject['box_points'][3]- eachObject['box_points'][1])
                if area>temp:
                    temp = area
                    array = eachObject['box_points']
    else:
        array = []


   
    if len(array) !=0:
        fig = plt.figure(figsize=(3, 3))
        
        sub_img = extract_patch_mask(img, array)
        results = model.detect([sub_img], verbose=1)
        r = results[0]
        mask = r['masks'][:, :, 0]
        rcnn_result = make_segmentation_mask(sub_img, mask)

        #img = cv2.resize(img, (960, 540)) 
        cv2.imshow('Original Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.rectangle(img, (array[0], array[1]), (array[2], array[3]), (0,0,255),3)
        cv2.imshow('Detection result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        list_intensities = find_pixels(rcnn_result, mask)


        bgr_arry = run_kmeans(list_intensities)
        # rgb_arry_init = np.zeros((1, 1, 3), dtype="uint8")
        # rgb_arry_init[0] = [r, g, b]
        # print ('bgr val', bgr_arry, type(bgr_arry))
        val = cl.label(bgr_arry[::-1])
        color_name = val['Generic Color Name']
        print('index', color_name)


        cv2.imshow('Segmentation result', rcnn_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.putText(rcnn_result, str(color_name), (600, 500), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 2)
        cv2.imshow('Final result',rcnn_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print ('No car found')

















