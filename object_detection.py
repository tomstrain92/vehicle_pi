import cv2


class Detector:
    """ detector class to perform object detection on image"""

    def __init__(self):
        prototxt = "weights/MobileNet_deploy.prototxt"
        weights = "weights/MobileNetSSD_deploy.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(prototxt, weights)

    def labels_and_boxes(self, image):

        detections = self.detect(image)
        image = parse_detections(detections, image)

        return image


    def detect(self, image):
        """class to perform detection on in image"""
        image_resized = cv2.resize(image, (300,300))
        blob = cv2.dnn.blobFromImage(image_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        # Set to network the input blob
        self.net.setInput(blob)
        # Prediction of network
        detections = self.net.forward()

        return detections


def parse_detections(detections, image, thresh=0.5):

    cols = 300; rows = 300
    bboxes = []
    labels = []
    centres = []
    # For get the class and location of object detected,
    # There is a fix index for class, location and confidence
    # value in @detections array .
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Confidence of prediction
        if confidence > thresh:  # Filter prediction
            class_id = int(detections[0, 0, i, 1])  # Class label

            # Object location
            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop = int(detections[0, 0, i, 5] * cols)
            yRightTop = int(detections[0, 0, i, 6] * rows)

            # Factor for scale to original size of frame
            heightFactor = image.shape[0] / 300.0
            widthFactor = image.shape[1] / 300.0

            # Scale object detection to frame
            xLeftBottom = int(widthFactor * xLeftBottom)
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop = int(widthFactor * xRightTop)
            yRightTop = int(heightFactor * yRightTop)
            # Draw location of object
            image = cv2.rectangle(image, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 255, 0))
            classNames = {0: 'background',
                          1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                          5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                          10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
                          14: 'motorbike', 15: 'person', 16: 'pottedplant',
                          17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

            # Draw label and confidence of prediction in frame resized
            if class_id in classNames:
                label = classNames[class_id] + ": " + str(confidence)
                classLabel = classNames[class_id]
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                yLeftBottom = max(yLeftBottom, labelSize[1])
                image = cv2.rectangle(image, (xLeftBottom, yLeftBottom - labelSize[1]),
                              (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                              (255, 255, 255), cv2.FILLED)
                image = cv2.putText(image, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                bboxes.append([(xLeftBottom, yLeftBottom), (xRightTop, yRightTop)])
                labels.append(classLabel)
                                
                height = xRightTop - xLeftBottom
                width = yRightTop - yLeftBottom
                xCentre = xLeftBottom + int(height/2)
                yCentre = yLeftBottom + int(width/2)
                centre = (xCentre, yCentre) 
                image = cv2.circle(image, centre, 5, (0,0,255), 2)
                # append object centres
                centres.append(centre)
                

    return [image, labels, bboxes, centres]



# testing
# det = Detector()
# # load image of car
# image = cv2.imread("cars.jpeg")
# image = det.labels_and_boxes(image)
#
# cv2.imshow('image', image)
# cv2.waitKey(0)

