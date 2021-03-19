import argparse
import logging
import time
import cv2
import re
import csv
import numpy as np
from contextlib import redirect_stdout
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')

    parser.add_argument('--output_json', type=str, default="False",
                        help='for json output')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    fields = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee', 
                'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar', 'Background']
    #clear output file
    with open('out.csv', 'w') as csvfile: 
            # creating a csv dict writer object 
            csvwriter = csv.writer(csvfile)             
            csvwriter.writerow(fields)   
    
    while True:
        ret_val, image = cam.read()

        #logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        #logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        print(humans[0].body_parts)
        #print(str(humans[0].body_parts[0])+', '+ str(humans[0].body_parts[1])) # {0: [0.54, 0.69], 1: [0.50, 0.92], 14: [0.50, 0.64], 15: [0.58, 0.64], 16: [0.44, 0.68], 17: [0.62, 0.68]}
        #print(humans[0].body_parts.keys()) # dict_keys([0, 14, 15, 16, 17])
        #print(humans[0].body_parts.items()) # dict_items([(0, BodyPart:0 x:0.56 y:0.64), (14, BodyPart:14 x:0.49 y:0.57), (15, BodyPart:15 x:0.60 y:0.55), (16, BodyPart:16 x:0.44 y:0.60), (17, BodyPart:17 x:0.64 y:0.55)])
        

        fields = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar', 'Background']
        #fields = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']
        #logger.debug('show+')
        body_list = [[str(humans[0].body_parts[0]), str(humans[0].body_parts[1]), str(humans[0].body_parts[2]), 
                    str(humans[0].body_parts[3]), str(humans[0].body_parts[4]), str(humans[0].body_parts[5]), 
                    str(humans[0].body_parts[6]), str(humans[0].body_parts[7]), str(humans[0].body_parts[8]), 
                    str(humans[0].body_parts[9]), str(humans[0].body_parts[10]), str(humans[0].body_parts[11]), 
                    str(humans[0].body_parts[12]), str(humans[0].body_parts[13]), str(humans[0].body_parts[14]), 
                    str(humans[0].body_parts[15]), str(humans[0].body_parts[16]), str(humans[0].body_parts[17])]] 
        
        with open('out.csv', 'a') as csvfile: 
            # creating a csv dict writer object 
            csvwriter = csv.writer(csvfile)            
            # writing data rows 
            csvwriter.writerows(body_list)

        '''
        try:
            if len(humans[0].body_parts.keys()) == 18: #wait for all joints to be in frame 
                with open(filename, 'w') as csvfile: 
                    # creating a csv dict writer object 
                    writer = csv.DictWriter(csvfile, fieldnames = fields) 
                      
                    # writing headers (field names) 
                    writer.writeheader() 
                      
                    # writing data rows 
                    writer.writerows(body_list)
            else:
                raise Exception("Need to get all joints in frame")
        except:
            print("No human in frame")
        '''
            
        '''
        try:
            if len(humans[0].body_parts.keys()) == 18: #wait for all joints to be in frame 
                with open('out.txt', 'a') as f:
                    with redirect_stdout(f):
                        print(humans[0].body_parts)
            else:
                raise Exception("Need to get all joints in frame")
        except:
            print("No human in frame")
        '''
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        #logger.debug('finished+')

    cv2.destroyAllWindows()
