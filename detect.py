from imageai.Detection import ObjectDetection
import os
import sys
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
import matplotlib.patches as patches
import kmedoids
ROOT_DIR = os.path.abspath("../")
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
import mrcnn.model as modellib
import coco


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

file_name = 'c_h_1.jpg'
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

        # print(eachObject["name"] , " : " , eachObject["percentage_probability"] )


def preprocess(rgb):
    blurred = cv2.GaussianBlur(rgb, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    #   print ('gray_shape', gray.shape)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                          cv2.THRESH_BINARY, 11, 2)

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
    kmeans = KMeans(n_clusters=3, max_iter=10000)
    kmeans.fit(array)

    unique_l, counts_l = np.unique(kmeans.labels_, return_counts=True)

    sort_ix = np.argsort(counts_l)
    sort_ix = sort_ix[::-1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_from = 0.05

    for cluster_center in kmeans.cluster_centers_[sort_ix]:
        print('point', int(cluster_center[2]), int(cluster_center[1]), int(cluster_center[0]))
        ax.add_patch(patches.Rectangle((x_from, 0.05), 0.29, 0.9, alpha=None,
                                       facecolor='#%02x%02x%02x' % (
                                       int(cluster_center[2]), int(cluster_center[1]), int(cluster_center[0]))))
        x_from = x_from + 0.31

    plt.show()


if len(array) !=0:
    sub_img, mask = extract_patch_mask(img, array)
    cv2.imwrite('test.jpg', sub_img)
    binary = preprocess(sub_img)
    cnts = find_countours(binary)
    lst_intensities, mask = draw_countours(cnts, sub_img, mask)
    cv2.imshow('', sub_img)
    cv2.waitKey(0)
    mul = mask * sub_img
    cv2.imshow('', mul)
    cv2.waitKey(0)
    run_kmeans(lst_intensities)
else:
    print ('No car found')


















