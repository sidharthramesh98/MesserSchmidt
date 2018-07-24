import cv2
import numpy as np
from utils import detector_utils
import tensorflow as tf
import sys
import cv2

 
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
detection_graph, sess = detector_utils.load_inference_graph()
sess = tf.Session(graph=detection_graph)
width = 680
height = 440
flago = False

def boundingbox(imgcv,width,height,detection_graph,sess):
    boxes,scores = detector_utils.detect_objects(imgcv,detection_graph,sess)
    print("boxes {}".format(boxes.shape))
    a = detector_utils.return_boxes(num_hands_detect = 2, score_thresh = 0.35, scores= scores, boxes = boxes, im_width = width, im_height = height, image_np = imgcv)
	# cv2.waitkey(0)
    print(a)
    if a != None:
        flag = True
        cv2.rectangle(imgcv,(a[0],a[1]),(a[2],a[3]),(0,255,0),2)
        cv2.imshow('frame',imgcv)
        return True,((a[0],a[1]),(a[2],a[3]))
    else:
        return False    
        
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[2]
 
if int(minor_ver) < 3:
    tracker = cv2.Tracker_create(tracker_type)
else:
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
   	    tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
 
# Read video
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

 
# Exit if video not opened.
if not video.isOpened():
    print "Could not open video"
    sys.exit()
 
# Read first frame.
# ok, frame = video.read()
# frame = cv2.flip(frame, 1)
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# # if not ok:
#     print 'Cannot read video file'
#     sys.exit()
     
# Define an initial bounding box
#bbox = boundingbox(frame)
# flago,bbox = boundingbox(frame,width,height,detection_graph,sess,flago)
# cv2.rectangle(frame,bbox[0],bbox[1],(0,255,0),2)
# #cv2.circle(frame,(int(bbox[0][0]-bbox[1][0]),int(bbox[0][1]-bbox[1][1])),10,(255,255,255),-11)
# cv2.imshow("Tracking", frame)

# print bbox
# xtl = bbox[1][0]
# ytl = bbox[1][1]
# w = np.abs([1][0]-bbox[0][0])
# h = np.abs(bbox[1][1]-bbox[0][1])
# print xtl,ytl,w,h

# ok = tracker.init(frame,(xtl,ytl,w,h))
 
while True:
        # Read a new frame
    ok, frame = video.read()
    frame = cv2.flip(frame, 1)
    if ok:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not ok:
        break
         
        # Start timer
    timer = cv2.getTickCount()
 
        # Update tracker
    ok, bbox = tracker.update(frame)
 
        # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
    if ok:
            # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else :
            # Tracking failure
        #flago = False
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        flago,bbox = boundingbox(frame,width,height,detection_graph,sess)
        if flago:    
            xtl = int(bbox[1][0])
            ytl = int(bbox[1][1])
            w = np.abs(int(bbox[1][0])-int(bbox[0][0]))
            h = np.abs(int(bbox[1][1])-int(bbox[0][1]))
            ok =tracker.init(frame,(xtl,ytl,w,h))
        else:
            continue
        # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
    cv2.imshow("Tracking", frame)
 
        # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break











# imgcv = cv2.imread("./sample_img/sample_person.jpg")
# result = tfnet.return_predict(imgcv)

# for i in range(len(result)):
# 	x,y = result[i]['bottomright']['x'],result[i]['bottomright']['y']
# 	xw,yh = result[i]['topleft']['x'], result[i]['topleft']['y']
# 	print (x,y,xw,yh)
# 	cv2.rectangle(imgcv,(x,y),(xw,yh),(0,255,0),2)
# cv2.imshow('frame',imgcv)
# cv2.waitKey(0)
# print(result)

