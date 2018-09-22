from imageai.Detection import ObjectDetection
import os
import sys
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
import matplotlib.patches as patches
from sklearn.mixture import GMM
from colorlabeler import ColorLabeler

import warnings
warnings.filterwarnings("ignore")

cl = ColorLabeler()


ROOT_DIR = os.path.abspath(os.getcwd())

sys.path.append(ROOT_DIR)
import mrcnn.model as modellib

sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
import coco

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
#config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

file_name = '1G1FF3D77H0120089_1.jpg'
execution_path = os.getcwd()


lst_intensities = []


detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, file_name),
                                             output_image_path=os.path.join(execution_path, "imagenew.jpg"))

img = cv2.imread(os.path.join(execution_path, file_name))



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
    mask = np.zeros_like(sub_img)
    return sub_img, mask


def draw_countours(cnts, sub_img, mask):
    lst_intensities = []

    for c in cnts:

        c = c.astype("float")
        c *= 1
        c = c.astype("int")
        cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
        pts = np.where(mask == 255)
        mask[mask == 255] = 1
        lst_intensities.append(sub_img[pts[0], pts[1]])
    return lst_intensities, mask


def run_kmeans(lst_intensities):

    array = np.array(lst_intensities[0])

    kmeans = KMeans(init='k-means++',n_clusters=3,random_state=0)
    kmeans.fit(array)
    gmm = GMM(n_components=3).fit(array)

    #print (gmm.cluster_centers_)
    labels = gmm.predict(array)
    kmeans_labels = kmeans.predict(array)

    unique_l, counts_l = np.unique(kmeans.labels_, return_counts=True)
    print ('kmeans', unique_l, counts_l)
    sort_ix = np.argsort(counts_l)
    print('sort_ix', sort_ix)
    sort_ix = sort_ix[::-1]
    print('rev', sort_ix)

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
    #
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


if len(array) !=0:
    sub_img, mask1 = extract_patch_mask(img, array)
    results = model.detect([sub_img], verbose=1)
    r = results[0]
    mask = r['masks'][:, :, 0]
    rcnn_result = make_segmentation_mask(sub_img, mask)
    # cv2.imshow('r', rcnn_result)
    # cv2.waitKey(0)
    binary = preprocess(rcnn_result)
    cv2.imshow('', binary)
    cv2.waitKey(0)
    binary[binary==255] = 1
    rcnn_result[:,:,0] *= binary
    rcnn_result[:,:,1] *= binary
    rcnn_result[:,:,2] *= binary
    cv2.imshow('r', rcnn_result)
    cv2.waitKey(0)
    list_intensities = find_pixels(rcnn_result, binary)
    # print (list_intensities)
    # for i in list_intensities:
    #     print (len(i))

    bgr_arry = run_kmeans(list_intensities)
    # rgb_arry_init = np.zeros((1, 1, 3), dtype="uint8")
    # rgb_arry_init[0] = [r, g, b]
    print ('bgr val', bgr_arry, type(bgr_arry))
    val = cl.label(bgr_arry[::-1])
#    print ('r,g,b', r,g,b)
    print('index', val)
else:
    print ('No car found')

















