# USAGE
# To read and write back out to video:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 \
#	--output output/output_01.avi
#
# To read from webcam and write back out to disk:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
#	--output output/webcam_output.avi
import numpy

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from PIL import Image
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import requests
from ultralytics import YOLO

# construct the argument parse and parse the arguments
'''
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
	help="# of skip frames between detections")
args = vars(ap.parse_args())
'''
model_cfg = 'yolo/yolov3-tiny.cfg'
model_wgt = 'yolo/yolov3-tiny.weights'
# model_cfg='/home/ncair/Downloads/people-counting-opencv/yolov3-coco/frozen_tiny_yolo_v3.xml'
# model_wgt='/home/ncair/Downloads/people-counting-opencv/yolov3-coco/frozen_tiny_yolo_v3.bin'

# input='/home/ncair/Desktop/Counter/people.mp4'
input = "rtsp://admin:transit@123@10.185.151.213/"
input = "/home/nms/people-counting/videos/hghh.3gp"
input = '../data/bskt.mp4'
# input = '../data/dnc.mp4'
# input = '../data/bolt.mp4'
REID_MODE = True
CONFIDENCE = 0.3
NMS_THRESHOLD = 0.3
skip_frames = 10
labels = ("person", "bicycle", "car", "motorbike", "aeroplane",
          "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird",
          "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack",
          "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat",
          "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
          "wine glass", "cup", "fork", "knife", "spoon",
          "bowl", "banana", "apple", "sandwich", "orange",
          "broccoli", "carrot", "hot dog", "pizza", "donut",
          "cake", "chair", "sofa", "pottedplant", "bed",
          "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven",
          "toaster", "sink", "refrigerator", "book", "clock",
          "vase", "scissors", "teddy bear", "hair drier", "toothbrush")

# lables = open("/home/nms/people-counting/yolo/coco-labels").read().strip().split('\n')

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromDarknet(model_cfg, model_wgt)
# net = cv2.dnn.readNet(model_cfg, model_wgt)
# net = cv2.dnn.readNetFromTensorflow('yolov3-coco/frozen_tiny_yolo_v3.pb')
#
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
layer_names = net.getLayerNames()
layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
# if a video path was not supplied, grab a reference to the webcam

vs = cv2.VideoCapture(input)

# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0


def detect_with_ultralytics(frame):
    # Load a model
    model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    results = model(frame)  # return a list of Results objects
    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs

        cls_tensor = boxes.cls
        confidence_tensor = boxes.conf
        bbox_tensor = boxes.xyxy

        get_item_func = lambda var: var.item()
        bbox_list = []
        for bbox in bbox_tensor:
            # bbox_list.append(list(map(get_item_func, list(bbox))))
            detection = list(bbox)
            box = detection[0:4] * np.array([1., 1., 1., 1.])
            bbox_list.append(box.astype('int'))
        confidence_list = list(map(get_item_func, list(confidence_tensor)))
        cls_list = list(map(get_item_func, list(cls_tensor)))
        # masks = result.masks  # Masks object for segmentation masks outputs
        # keypoints = result.keypoints  # Keypoints object for pose outputs
        # probs = result.probs  # Probs object for classification outputs
        # obb = result.obb  # Oriented boxes object for OBB outputs

        # result.show()  # display to screen
        # result.save(filename="result.jpg")  # save to disk

    # return ordered_results
    return bbox_list, confidence_list, cls_list


def generate_boxes_confidences_classids(outs, height, width, tconf):
    boxes = []
    confidences = []
    classids = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            if confidence > CONFIDENCE:
                if labels[classid] != "person":
                    continue
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))
                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)
    return boxes, confidences, classids


# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
fr_adj = 1
while True:

    fr_adj += 1
    if fr_adj % 1 == 0:
        totalFrames += 1
        # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        grab, frame = vs.read()
        if grab == True:
            frame = frame  # if args.get("input", False) else frame
            # frame = frame[100:500, 250:704]
            # print(frame.shape[1])
            # if we are viewing a video and we did not grab a frame then we
            # have reached the end of the video

            # resize the frame to have a maximum width of 500 pixels (the
            # less data we have, the faster we can process it), then convert
            # the frame from BGR to RGB for dlib

            # frame = imutils.resize(frame, width=350)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # if the frame dimensions are empty, set them
            if W is None or H is None:
                (H, W) = frame.shape[:2]
            height, width = frame.shape[:2]

            # if we are supposed to be writing a video to disk, initialize
            # the writer
            '''
            if args["output"] is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(args["output"], fourcc, 30,
                    (W, H), True)
            '''
            # initialize the current status along with our list of bounding
            # box rectangles returned by either (1) our object detector or
            # (2) the correlation trackers
            status = "Waiting"
            rects = []

            # check to see if we should run a more computationally expensive
            # object detection method to aid our tracker
            if totalFrames % skip_frames == 0:
                # set the status and initialize our new set of object trackers
                status = "Detecting"
                trackers = []

                boxes, confidences, classids = detect_with_ultralytics(frame)

                idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, NMS_THRESHOLD)
                idxs = range(len(boxes))
                count = 0
                if len(idxs) > 0:
                    # for i in idxs.flatten():
                    for i in idxs:
                        count += 1
                        print(labels[int(classids[i])])
                        x, y = boxes[i][0], boxes[i][1]
                        w, h = boxes[i][2], boxes[i][3]

                        # if confidences[i] > CONFIDENCE:
                        #     if labels[int(classids[i])] != "person":
                        #         continue
                        if REID_MODE:
                            cropped_image = frame[y:h, x:w]
                            # cv2.imwrite(f'gallery/dets/sample1_{x}_{y}_{totalFrames}.jpg', cropped_image)
                            cv2.imwrite(f'gallery/dets/sample1_{totalFrames}_{count}_{labels[int(classids[i])]}.jpg',
                                        cropped_image)
                            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                        else:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # construct a dlib rectangle object from the bounding
                        # box coordinates and then start the dlib correlation
                        # tracker
                        tracker = dlib.correlation_tracker()
                        # rect = dlib.rectangle(x, y, x + w, y + w)
                        rect = dlib.rectangle(x, y, w, h)
                        tracker.start_track(rgb, rect)

                        # add the tracker to our list of trackers so we can
                        # utilize it during skip frames
                        trackers.append(tracker)

            # otherwise, we should utilize our object *trackers* rather than
            # object *detectors* to obtain a higher frame processing throughput
            else:
                # loop over the trackers
                for tracker in trackers:
                    # set the status of our system to be 'tracking' rather
                    # than 'waiting' or 'detecting'
                    status = "Tracking"

                    # update the tracker and grab the updated position
                    tracker.update(rgb)
                    pos = tracker.get_position()

                    # unpack the position object
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    # add the bounding box coordinates to the rectangles list
                    rects.append((startX, startY, endX, endY))

            # draw a horizontal line in the center of the frame -- once an
            # object crosses this line we will determine whether they were
            # moving 'up' or 'down'
            # TODO: removed lines
            # cv2.line(frame, (0, int(H // 4)), (W, int(H // 4)), (0, 255, 255), 1)
            # cv2.line(frame, (0, int(H // 1.25)), (W, int(H // 1.25)), (0, 0, 255), 1)

            # use the centroid tracker to associate the (1) old object
            # centroids with (2) the newly computed object centroids

            objects = ct.update(rects, frame)

            # loop over the tracked objects
            for (objectID, centroid) in objects.items():
                # check to see if a trackable object exists for the current
                # object ID
                to = trackableObjects.get(objectID, None)

                # if there is no existing trackable object, create one
                if to is None:
                    to = TrackableObject(objectID, centroid)

                # otherwise, there is a trackable object so we can utilize it
                # to determine direction
                else:
                    # the difference between the y-coordinate of the *current*
                    # centroid and the mean of *previous* centroids will tell
                    # us in which direction the object is moving (negative for
                    # 'up' and positive for 'down')
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)

                    # check to see if the object has been counted or not
                    if not to.counted:
                        # if the direction is negative (indicating the object
                        # is moving up) AND the centroid is above the center
                        # line, count the object
                        if direction < 0 and centroid[1] < H // 2:
                            totalUp += 1
                            # data="http://18.191.211.159/counter.php?Store_Id="+str(2)+"&upcount="+str(totalUp)+"&downcount="+str(totalDown)
                            # r = requests.get(url = data)
                            # print(r)
                            to.counted = True

                        # if the direction is positive (indicating the object
                        # is moving down) AND the centroid is below the
                        # center line, count the object
                        elif direction > 0 and centroid[1] > H // 2:
                            totalDown += 1
                            # data="http://18.191.211.159/counter.php?Store_Id="+str(2)+"&upcount="+str(totalUp)+"&downcount="+str(totalDown)
                            # r = requests.get(url = data)
                            # print(r)
                            to.counted = True

                # store the trackable object in our dictionary
                trackableObjects[objectID] = to

                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            # construct a tuple of information we will be displaying on the
            # frame
            info = [
                ("Up", totalUp),
                ("Down", totalDown),
                ("Status", status),
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # check to see if we should write the frame to disk
            if writer is not None:
                writer.write(frame)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            # increment the total number of frames processed thus far and
            # then update the FPS counter

            fps.update()

# stop the timer and display FPS information


cv2.destroyAllWindows()
