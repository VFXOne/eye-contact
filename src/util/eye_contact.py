"""Utility method to draw the eye contact"""
import cv2 as cv
import numpy as np

def draw_contact(image_in, face, gaze_vector):
    image_out = image_in
    if (is_contact(gaze_vector)): #Draw green
        cv.rectangle(
                    image_out, tuple(np.round(face[:2]).astype(np.int32)),
                    tuple(np.round(np.add(face[:2], face[2:])).astype(np.int32)),
                    color=(0, 255, 0), thickness=1, lineType=cv.LINE_AA,
                )
    else: #Draw red
        cv.rectangle(
                    image_out, tuple(np.round(face[:2]).astype(np.int32)),
                    tuple(np.round(np.add(face[:2], face[2:])).astype(np.int32)),
                    color=(0, 0, 255), thickness=1, lineType=cv.LINE_AA,
                )
    return image_out
    
            
def is_contact(gaze_vector):
    #The eye contact is defined as looking within a 5° range around the camera
    max_angle = np.radians(5)

    phi = np.absolute(gaze_vector[1])
    theta = np.absolute(gaze_vector[0])
    
    return phi < max_angle and theta < max_angle
