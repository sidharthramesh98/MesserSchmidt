from utils import detector_utils as detector_utils 
from utils import Net as Net
from utils import track as track
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool
import time
from utils.detector_utils import WebcamVideoStream
import datetime
import argparse

frame_processed = 0
score_thresh = 0.35
num_hands_detect = 2

def is_empty(any_structure):
    if any_structure:
        #print('Structure is not empty.')
        return False
    elif any_structure==None:
        return False
    else:
        #print('Structure is empty.')
        return True


# Create a worker thread that loads graph and
# does detection on images in an input queue and puts it on an output queue

def worker(input_q, output_q, cap_params, frame_processed):
    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    count = 1
    list2 = []
    tracker = track.tracker_init()
    while True:
        print("Loop running")
        if is_empty(list2):
            print("In if")
            frame = input_q.get()
            print("ckpt 1")
            boxes, scores = detector_utils.detect_objects(frame, detection_graph, sess)
            print("ckpt 2")
            list2 = detector_utils.return_boxes(cap_params['num_hands_detect'], cap_params["score_thresh"], scores, boxes, cap_params['im_width'], cap_params['im_height'], frame)
            print("ckpt 3")
            print("list2:{}".format(list2))
            if not is_empty(list2):    
                box = list2
                xtl = box[0]
                ytl = box[1]
                w = np.abs(box[2]-box[0])
                h = np.abs(box[3]-box[1])
                ok = tracker.init(frame,(xtl,ytl,w,h))
                print(ok)
                continue
            #continue
            else :
                 print("list is empty")
                 continue
        #print("> ===== in worker loop, frame  frame_processed)
        else :
            print("In else")
            frame = input_q.get()
            print(frame.shape)
            if (frame is not None):
                # actual detection
                boxes, scores = detector_utils.detect_objects(
                    frame, detection_graph, sess)
                # draw bounding boxes
                print("boxes {}".format(boxes.shape))
                detector_utils.export_boxes(cap_params['num_hands_detect'], cap_params["score_thresh"], scores, boxes, cap_params['im_width'], cap_params['im_height'], frame)
                #if count == 1 or list2 == None or list2 == [0]:
                #    list2 = detector_utils.return_boxes(cap_params['num_hands_detect'], cap_params["score_thresh"], scores, boxes, cap_params['im_width'], cap_params['im_height'], frame) 
                detector_utils.draw_box_on_image(
                    cap_params['num_hands_detect'], cap_params["score_thresh"], scores, boxes, cap_params['im_width'], cap_params['im_height'], frame)
                # add frame annotated with bounding box to queue
                print("list2 {}".format(list2))
                #if list2 != None:
                track.run_tracker(tracker = tracker,frame = frame, box = list2)
                output_q.put(frame)
                frame_processed += 1
                count += 1
            else:
                output_q.put(frame)
    sess.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-nhands', '--num_hands', dest='num_hands', type=int,
                        default=2, help='Max number of hands to detect.')
    parser.add_argument('-fps', '--fps', dest='fps', type=int,
                        default=1, help='Show FPS on detection/display visualization')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=680, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=440, help='Height of the frames in the video stream.')
    parser.add_argument('-ds', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=1, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=7, help='Size of the queue.')
    args = parser.parse_args()

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)

    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()

    cap_params = {}
    frame_processed = 0
    cap_params['im_width'], cap_params['im_height'] = video_capture.size()
    cap_params['score_thresh'] = score_thresh

    # max number of hands we want to detect/track
    cap_params['num_hands_detect'] = args.num_hands

    print(cap_params, args)

    # spin up workers to paralleize detection.
    try :
        pool = Pool(args.num_workers, worker,
                    (input_q, output_q, cap_params, frame_processed))

        start_time = datetime.datetime.now()
        num_frames = 0
        fps = 0
        index = 0
        #print("input {}".format(list2))
        while True:
            frame = video_capture.read()
            frame = cv2.flip(frame, 1)
            index += 1

            print("Input frame")
            input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            print("Term input")
            print(output_frame)
            output_frame = output_q.get()
            print("Output frame")
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

            elapsed_time = (datetime.datetime.now() -
                            start_time).total_seconds()
            num_frames += 1
            fps = num_frames / elapsed_time
            # print("frame ",  index, num_frames, elapsed_time, fps)

            if (output_frame is not None):
                if (args.display > 0):
                    if (args.fps > 0):
                        detector_utils.draw_fps_on_image(
                            "FPS : " + str(int(fps)), output_frame)
                    cv2.imshow('Muilti - threaded Detection', output_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    if (num_frames == 400):
                        num_frames = 0
                        start_time = datetime.datetime.now()
                    else:
                        print("frames processed: ",  index,
                              "elapsed time: ", elapsed_time, "fps: ", str(int(fps)))
            else:
                # print("video end")
                break
        elapsed_time = (datetime.datetime.now() -
                        start_time).total_seconds()
        fps = num_frames / elapsed_time
        print("fps", fps)
        pool.terminate()
        video_capture.stop()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        sys.exit()
