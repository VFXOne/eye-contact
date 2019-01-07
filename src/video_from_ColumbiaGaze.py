import skvideo.io as skv
from PIL import Image
import numpy as np
import csv
import os
import random
import argparse
from math import radians

def filename_to_theta_phi(filename):
    args = filename.split('_')
    camera_distance = int(args[1][:-1])
    camera_pos = int(args[2][:-1])
    vertical_angle = int(args[3][:-1])
    horizontal_angle = int(args[4][:-5])
    phi = radians(horizontal_angle)
    theta = radians(vertical_angle)
    return (theta, phi)


if __name__ == '__main__':
    # Set arguments
    parser = argparse.ArgumentParser(description='Create a video from Columbia Gaze dataset')
    parser.add_argument('--number_of_frames', type=int, required=True, help='Specify the number of images to be taken for creating the video')
    parser.add_argument('--path', type=str, default='Columbia Gaze Data Set', help='Specify the path where to find the dataset')
    parser.add_argument('--annotate_lines', type=bool, default=False, help='Annotate with "phi" and "theta" the corresponding angles int the annotation file')

    args = parser.parse_args()

    assert os.path.isdir(args.path)

    output_video_name = "columbia_video.mp4"
    #video_out = VideoWriter(output_video_name, frameSize = (1270,720))
    #video_out.open()
    video_writer = skv.FFmpegWriter(output_video_name, 
                                    inputdict={'-r': str(2)},
                                    outputdict={'-r': str(2)})
    #image_array = np.zeros(((args.number_of_frames, 720, 1280, 3)))
    annotation_file_name = "angle_data.txt"
    angle_file = open(annotation_file_name, 'w')

    # Define some constants for the MPIIGaze dataset
    participant_number = 56

    for i in range(args.number_of_frames):
        participant = random.randint(1,participant_number)
        p_path = args.path + '/{:04}/'.format(participant)
        number_images = len(os.listdir(p_path)) - 1
        image_number = random.randint(1, number_images)
        image_name = os.listdir(p_path)[image_number]
        if len(image_name) < 15:
            i-=1
            continue
        #resize the image
        basewidth = 1280
        img = Image.open(p_path + image_name)
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth,hsize), Image.ANTIALIAS)
        img.save('resized.jpg') 
        frame = skv.vread('resized.jpg')
        os.remove('resized.jpg')
        #write the image in video
        video_writer.writeFrame(np.array(frame))
        theta, phi = filename_to_theta_phi(image_name)
        with open(annotation_file_name, 'a+') as f:
            if args.annotate_lines:
                f.write('Frame index: {} Phi: {} Theta: {}\n'.format(i, phi, theta))
            else:
                f.write('{} {} {}\n'.format(i, phi, theta))
        print('\rProgress: {:.1f}%'.format(i/args.number_of_frames*100), end = '\r')

    #skv.vwrite(output_video_name, image_array)
    video_writer.close()

    print('The output video is {} and the annotation file {}'.format(output_video_name, annotation_file_name))

        