import skvideo.io as skv
from PIL import Image
import numpy as np
import csv
import os
import random
import argparse
from math import atan2, acos, sqrt

def toPhiTheta(x, y, z):
    return (atan2(-x, -z), acos(z/(sqrt(x*x + y*y + z*z))))

if __name__ == '__main__':
    # Set arguments
    parser = argparse.ArgumentParser(description='Create a video from MPIIGaze dataset')
    parser.add_argument('--number_of_frames', type=int, required=True, help='Specify the number of images to be taken for creating the video')
    parser.add_argument('--path', type=str, default='MPIIGaze', help='Specify the path where to find the dataset')    
    parser.add_argument('--annotate_lines', type=bool, default=True, help='Annotate with "phi" and "theta" the corresponding angles int the annotation file')

    args = parser.parse_args()

    output_video_name = "output_video.mp4"
    #video_out = VideoWriter(output_video_name, frameSize = (1270,720))
    #video_out.open()
    video_writer = skv.FFmpegWriter(output_video_name)
    #image_array = np.zeros(((args.number_of_frames, 720, 1280, 3)))
    annotation_file_name = "angle_data.txt"
    angle_file = open(annotation_file_name, 'w')

    # Define some constants for the MPIIGaze dataset
    participant_number = 14

    for i in range(args.number_of_frames):
        participant = random.randint(1,participant_number)
        p_path = args.path + '/Data/Original/p{:02}/'.format(participant)
        days_for_participant = len(os.listdir(p_path)) - 1
        day = random.randint(1,days_for_participant)
        d_path = p_path + "day{:02}".format(day)
        number_images = len(os.listdir(d_path)) - 1
        image_number = random.randint(1, number_images)
        if image_number < 2:
            i= i -1
            continue
        
        frame = skv.vread(d_path + '/' + '{:04}.jpg'.format(image_number))
        #image_array[i,:,:,:] = np.array(frame)
        video_writer.writeFrame(np.array(frame))
        annotation_file = open(d_path + '/annotation.txt', 'r')
        csv_reader = csv.reader(annotation_file, delimiter=' ')
        next(csv_reader, None)
        line = list(csv_reader)[image_number-2]
        x = line[26]
        y = line[27]
        z = line[28]
        phi, theta = toPhiTheta(float(x), float(y), float(z))
        if args.annotate_lines:
            angle_file.write('Frame index: {} Phi: {} Theta: {}\n'.format(i, phi, theta))
        else :
            angle_file.write(str(i) + ' ' + str(phi) + ' ' + str(theta) + '\n')
        print('\rProgress: {:.1f}%'.format(i/args.number_of_frames*100), end = '\r')
    #skv.vwrite(output_video_name, image_array)
    video_writer.close()

    print('The output video is {} and the annotation file {}'.format(output_video_name, annotation_file_name))

        