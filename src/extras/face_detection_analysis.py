from datasources import Video
from models import ELG

import tensorflow as tf
import cv2 as cv
import numpy as np
import os

#input = 'columbia_video_500.mp4'
input = 'me.jpg'
temp_frame = 'temp_frame.png'
cap = cv.VideoCapture(input)

from tensorflow.python.client import device_lib
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

tf.logging.set_verbosity(tf.logging.INFO)
with tf.Session(config=session_config) as session:

    data_source = Video(input, tensorflow_session=session, 
                        batch_size=2,
                        data_format='NHWC',
                        eye_image_shape=(108, 180))
    
    model = ELG(
                session, train_data={'videostream': data_source},
                first_layer_stride=3,
                num_modules=3,
                num_feature_maps=64,
                learning_schedule=[
                    {
                        'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
                    },
                ],
            )

    while(True):

        ret, image = cap.read()
                
        if image is None:
            print('End of video')
            raise SystemExit
        
        #data_source.frame_read_job()
        #before_frame_read, bgr, after_frame_read = data_source._frame_read_queue.get()
        bgr = image
        current_index = 1
        grey = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        
        frame = {
            'frame_index': current_index,
            'time': {
                'before_frame_read': 0,
                'after_frame_read': 0,
            },
            'bgr': bgr,
            'grey': grey,
        }
        
        #data_source.detect_faces(frame)
        frame['faces'] = [[463,55,167,208]]
        data_source.detect_landmarks(frame)
        data_source.calculate_smoothed_landmarks(frame)
        data_source.update_face_boxes(frame)

        face = frame['faces']
        
        print(face)
        
        frame_landmarks = (frame['smoothed_landmarks']
                           if 'smoothed_landmarks' in frame
                           else frame['landmarks'])

        for f, face in enumerate(frame['faces']):
            for landmark in frame_landmarks[f][:-1]:
                cv.drawMarker(bgr, tuple(np.round(landmark).astype(np.int32)),
                              color=(0, 0, 255), markerType=cv.MARKER_STAR,
                              markerSize=2, thickness=1, line_type=cv.LINE_AA)
            cv.rectangle(
                bgr, tuple(np.round(face[:2]).astype(np.int32)),
                tuple(np.round(np.add(face[:2], face[2:])).astype(np.int32)),
                color=(0, 255, 255), thickness=1, lineType=cv.LINE_AA,
            )

        cv.imshow('vis', bgr)
        cv.waitKey(0)
        #cv.imwrite('frame.jpg', bgr) 
    
cap.release()

