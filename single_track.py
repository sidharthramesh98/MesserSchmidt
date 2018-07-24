from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
from utils import track as track
import datetime
import argparse

#cap = cv2.VideoCapture(0)
detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float,
                        default=0.75, help='Score threshold for displaying bounding boxes')
    parser.add_argument('-fps', '--fps', dest='fps', type=int,
                        default=1, help='Show FPS on detection/display visualization')
    parser.add_argument('-src', '--source', dest='video_source',
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=680, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=440, help='Height of the frames in the video stream.')
    parser.add_argument('-ds', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=1, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)
    print("starting")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2

    list2 = [0]
    count = 1
    tracker = track.tracker_init()
    flag = True
    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()

        image_np = cv2.flip(image_np, 1)
        if ret == True:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
        else:
            print("error")

        # actual detection
        boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess)
        # draw bounding boxes
        detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh, scores, boxes, im_width, im_height, image_np)
        if count == 1 or list2 == None or list2 == [0] or flag == False:
            list2 = detector_utils.return_boxes(num_hands_detect, args.score_thresh, scores, boxes, im_width, im_height, image_np)
            print(flag)
            print("list2 changed")
        if list2 != None:
            flag = track.run_tracker(tracker = tracker,frame = image_np, box = list2)
        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() -
                        start_time).total_seconds()
        fps = num_frames / elapsed_time

        count = count+1
        if (args.display > 0):
            # Display FPS on frame
            if (args.fps > 0):
                detector_utils.draw_fps_on_image(
                    "FPS : " + str(int(fps)), image_np)

            cv2.imshow('Single Threaded Detection', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            print("frames processed: ",  num_frames,
                  "elapsed time: ", elapsed_time, "fps: ", str(int(fps)))
    cap.release()
    cv2.destroyAllWindows()