import csv 
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from util import eye_contact

if __name__ == '__main__':
    # Set arguments
    parser = argparse.ArgumentParser(description='Create a video from MPIIGaze dataset')
    parser.add_argument('--computed_file', type=str, required=True, help='Specify the file for the computed angles')
    parser.add_argument('--truth_file', type=str, required=True, help='Specify the file for theground truth')
    parser.add_argument('--save_errors', type=bool, default=False, help='Save the errors in a separate file "errors.txt"')
    parser.add_argument('--show_max_error', type=bool, default=True, help='Show the first 10 frames number with the higher relative error')
    parser.add_argument('--show_min_error', type=bool, default=True, help='Show the first 10 frames number with the lower relative error')

    
    args = parser.parse_args()
    
    assert os.path.isfile(args.computed_file)
    assert os.path.isfile(args.truth_file)
    
    computed_file = open(args.computed_file, 'r')
    truth_file = open(args.truth_file, 'r')
    
    #Open both file as CSV file
    csv_comp = csv.reader(computed_file, delimiter = ' ')
    csv_truth = csv.reader(truth_file, delimiter = ' ')
    
    comp_list = list(csv_comp)
    truth_list = list(csv_truth)
    frames_computed = len(comp_list)
    
    comp_count = 0
    
    theta_values = []
    theta_errors = []
    
    phi_values = []
    phi_errors = []

    
    associated_frame = []

    eye_contact_errors = []
    
    for truth_count in range(len(truth_list)):
    
        if frames_computed - comp_count <= 0: 
            break
        
        comp_frame = int(comp_list[comp_count][0])
        
        if (truth_count != comp_frame):
            continue

        #phi_comp must be flipped because the video is mirrored
        phi_comp_left = -float(comp_list[comp_count][1])
        theta_comp_left = float(comp_list[comp_count][2])
        phi_comp_right = None
        theta_comp_right = None

        #Must take into account both eyes if available
        if (comp_count < frames_computed-1 and int(comp_list[comp_count + 1][0]) == comp_frame): #Second eye angle available
            comp_count += 1
            phi_comp_right = -float(comp_list[comp_count][1])
            theta_comp_right = float(comp_list[comp_count][2])

        phi_truth = float(truth_list[truth_count][1])
        theta_truth = float(truth_list[truth_count][2])
        
        absolute_error_theta_left = theta_truth - theta_comp_left
        absolute_error_phi_left = phi_truth - phi_comp_left

        theta_values.append(theta_truth)
        theta_errors.append(absolute_error_theta_left)
        phi_values.append(phi_truth)
        phi_errors.append(absolute_error_phi_left)

        truth_eye_contact = eye_contact.is_contact([theta_truth, phi_truth])
        eye_contact_left = eye_contact.is_contact([theta_comp_left, phi_comp_left])
        eye_contact_errors.append(truth_eye_contact == eye_contact_left)

        if (phi_comp_right is not None):
            absolute_error_theta_right = theta_truth - theta_comp_right
            absolute_error_phi_right = phi_truth - phi_comp_right
            eye_contact_right = eye_contact.is_contact([theta_comp_right, phi_comp_right])


            theta_values.append(theta_truth)
            theta_errors.append(absolute_error_theta_right)
            phi_values.append(phi_truth)
            phi_errors.append(absolute_error_phi_right)
            eye_contact_errors.append(truth_eye_contact == eye_contact_right)


        associated_frame.append(comp_frame)
        
        comp_count+=1
        
    computed_file.close()
    truth_file.close()

    #Print eye contact detection accuracy
    print('Eye contact accuracy: {:.04}%\n'.format(np.sum(eye_contact_errors)/len(eye_contact_errors)*100))
    
    plt.figure(1)
    plt.plot(theta_values, theta_errors, 'b.')
    plt.xlabel('Theta (radians)')
    plt.ylabel('Absolute error')
    
    plt.figure(2)
    plt.plot(phi_values, phi_errors, 'r.')
    plt.xlabel('Phi (radians)')
    plt.ylabel('Absolute error')
    
    plt.figure(3)
    plt.hist(phi_errors, bins = 'auto')
    plt.xlabel('Phi errors')
    
    plt.figure(4)
    plt.hist(theta_errors, bins = 'auto')
    plt.xlabel('Theta errors')
    
    plt.show()
    
    if (args.save_errors):
        with open("errors.txt", 'w') as f:
            f.write('phi angles: {}, phi errors {}\n'.format(phi_values, phi_errors))
            f.write('theta angles: {}, theta errors {}\n'.format(theta_values, theta_errors))
            f.write('Associated frame number: {}'.format(associated_frame))
    
    if args.show_max_error:
        sorted_phi = [x for _,x in sorted(zip(phi_errors,associated_frame))]
        sorted_theta = [x for _,x in sorted(zip(theta_errors,associated_frame))]
        # with open('frames_error.txt', 'w') as f:
        #     f.write('Worst Phi error {}\n'.format(sorted_phi)) 
        #     f.write('Worst Theta error {}\n'.format(sorted_theta)) 
        print('Worst Phi error {}\n'.format(sorted_phi[:10])) 
        print('Worst Theta error {}\n'.format(sorted_theta[:10])) 

    if args.show_min_error:
        print('Best Phi frames {}\n'.format(sorted_phi[-10:]))
        print('Best Theta frames {}\n'.format(sorted_theta[-10:]))
        
    if args.show_min_error:
        middle_phi = int(len(sorted_phi)/2)
        middle_theta = int(len(sorted_theta)/2)
        print('Average Phi frames {}\n'.format(sorted_phi[(middle_phi - 5):(middle_phi + 5)]))
        print('Average Theta frames {}\n'.format(sorted_theta[(middle_theta - 5):(middle_theta + 5)]))
    