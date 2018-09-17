from imageai.Detection import ObjectDetection
import os

import cv2

file_name = '19UDE2F32HA002792_1.jpg'
execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, file_name),
                                             output_image_path=os.path.join(execution_path, "imagenew.jpg"))

temp = 0
for eachObject in detections:
    if eachObject['name'] == 'car':
        if eachObject['percentage_probability'] > temp:
            temp = eachObject['percentage_probability']
            array = eachObject['box_points']

        # print(eachObject["name"] , " : " , eachObject["percentage_probability"] )


def preprocess(rgb):
    blurred = cv2.GaussianBlur(rgb, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    #   print ('gray_shape', gray.shape)
    binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    thresh = binary.max() - binary
    return thresh


def find_countours(binary_img):
    _, cnts, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    return cnts


x1, y1, x2, y2 = array[0], array[1], array[2], array[3]

img = cv2.imread(os.path.join(execution_path, file_name))

#cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

sub_img = img[y1:y2, x1:x2]

# print ('shape', sub_img.shape)
binary = preprocess(sub_img)

cv2.imshow('"', binary)
cnts = find_countours(binary)
print (len(cnts))
#print ('cns', cnts)
for c in cnts:
    # compu1te the center of the contour
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]))
    cY = int((M["m01"] / M["m00"]))

    # detect the shape of the contour and label the color

    #	color = cl.label(lab, c)

    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape and labeled
    # color on the image
    c = c.astype("float")
    c *= 1
    c = c.astype("int")
    #	text = "{}".format(color)
    cv2.drawContours(sub_img, [c], -1, (255, 0, 0), 2)
    #	cv2.putText(image, text, (cX, cY),
    #		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('', sub_img)
    cv2.waitKey(0)
