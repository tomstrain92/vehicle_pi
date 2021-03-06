# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
from object_detection import Detector
from tracking import *

def main():
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 30
    rawCapture = PiRGBArray(camera, size=(640, 480))

    # allow the camera to warmup
    time.sleep(0.1)

    # load Detector
    det = Detector()
    # list of objects
    objects = []
    objects.append(imageObject, "person", 50, 50)

    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        [image, labels, bboxes, centres] = det.labels_and_boxes(image)
        for label, bbox, centre in zip(labels, bboxes, centres):
            print(label, bbox, centre)
        # update objects
        objects = update_all_objects(objects, labels, centres)

        # show the frame
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
