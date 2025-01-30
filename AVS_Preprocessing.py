'''
Project: Aortic valve stenosis prediction with convolutional neural network using imaging data.
Description: This program processes video files in a given directory using several image processing techniques in
order to obtain the most important and relevant frames for further neural network training.
Author: B.Sc. Nazlıgül Keske
Supervisor: M.Sc. Annika Engel
Institution: Chair for Clinical Bioinformatics, Saarland University.
Contact: nake00002@stud.uni-saarland.de
Date: October 31, 2023


Directories:
-video_dir: Contains input imaging videos to be preprocessed in .avi format. Videos can be in various frame-per-second record setting.
-output_dir: Contains preprocessed videos in .avi format, resulted from the program. All output videos saved in 8 frame-per-second setting.
-'Important_frames: Contains imprortant frame images in .jpeg format extracted from each input video seperately in each iteration.
- reference_image: Contains referance location of the valve in JPG format to be used in segmentation

Functions:
-load_video_as_array():  Loads a video as an array of frames using OpenCV.
    Args:
    input_video_path (str): The path of the video file.
    Returns:
    numpy.ndarray: A numpy array representing the video.

-change_fps_down(): Changes the frame rate (fps) of a video by down-sampling the frames.
    It loads the video as an array using the function load_video_as_array, then calculates the current frame rate of the video (fps_old)
    using OpenCV's VideoCapture() function. Function operates down-sampling by excluding some frames in an ordered manner.
    This function called if the input video has higher fps than desired standard fps which is 8.
    Args:
    -input_video_path: (str) path to the input video file.
    -fps: (int) desired new frame rate of the output video.
    Returns:
    -new_video_array: (numpy.ndarray) A numpy array representing the modified video.

-change_fps_up():This function changes the frame rate of a video by either increasing it.
    The function first loads the video as a numpy array using the 'load_video_as_array' function, and then obtains the
    frame rate of the video using the 'cv2.VideoCapture' function. The function either copies frames directly to the new video array if it corresponds
    to a keyframe, or interpolates a new frame between the previous and next keyframe.
    This function called if the input video has lower fps than desired standard fps which is 8.
    Args:
    input_video_path (str): The path of the video file to be modified.
    fps (int): The new frame rate to be set.
    Returns:
    numpy.ndarray: A numpy array representing the modified video.

-black_object_detection(): This function is used when there is observable black stitches or objects that blocks the opening of the valve.
    In this function Canny edge detection is applied and the edges of the blocking object is extracted by hysteresis thresholding.
    (***Not automated, use case should be indicated in the webserver interface***)
    Args:
    image (2D numpy.ndarray): Image to extract the object from
    low_threshold (Hysteresis threshold 1) = 100
    high_threshold (Hysteresis threshold 2)= 190
    kernel_size = 45
    Returns:
    numpy.ndarray: A numpy array in binary form representing the pixels of blocking object

-replace_stitched_pixels(): This function replaces the gray value of pixels that includes blocking objects or stitches with the mean gray
    value of surrounding pixels.
    (***Not automated, use case should be indicated in the webserver interface***)
    Args:
    image (2D numpy.ndarray): Image to remove the object from
    binary_mask (2D numpy.ndarray): A numpy array in binary form representing the pixels of blocking object
    kernel_size=45
    Returns:
    numpy.ndarray: A numpy array containing gray values after object removal

-segment_valve_opening(): This function applies segmentation algorithm based on the frame differences calculated between 0th frame and
    contrast starting frame +5 to get the whole valve area. Then some sliding and Chi-squared shifting methods are applied accordinf to referance image.
    Args:
    image (3D numpy.ndarray): 3D numpy array image
    frame_boundry (int): Contrast starting frame calculated by algorithm.
    reference_image(JPG): Reference image used in chi-squared shifting.
    Returns:
    1D array: Dimentions and coordinates of segmentation box

-process_videos_in_dir(): This function takes a directory path where video files are stored and an output directory path
    where the important frames of each video are saved. Extracted important frames are saved to'Important_frames' directory to further create
    processes .avi files for each input file.
    Args:
    video_dir (str): The path to the directory containing the input video files.
    output_dir (str): The path to the directory where the processed output video file will be saved.
    Returns:
    None

* The program uses several image processing methods, including cropping, standardization, Gaussian blurring, Otsu thresholding,
morphological operations, and contour detection, to detect the most important frames in each input video in the context of aortic valve stenosis.
These image processing methods are applied with the corresponding functions in OpenCV libraby.

'''

import glob
from scipy.signal import argrelextrema
import matplotlib
import numpy as np
import cv2
import panel as pn
import os
from skimage import io
from image_registration import chi2_shift

matplotlib.use('TkAgg')
pn.extension()

# Defining the directory path that contains the video and reference image files
video_dir = 'Input_videos'
output_dir = 'Output_videos'
reference_image = io.imread("reference_image.jpg")


def run():
    process_videos_in_dir(video_dir, output_dir)


def load_video_as_array(input_video_path):
    vidcap = cv2.VideoCapture(input_video_path)
    ts = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    success, tmp = vidcap.read()
    count = 0
    image = []

    while success:
        frame = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        frame = np.array(frame)
        frame = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
        if success:
            image.append(frame)
        else:
            break

        success, tmp = vidcap.read()
        count = count + 1

    image = np.array(image)

    vidcap.release()
    cv2.destroyAllWindows()

    tmp_rot = np.swapaxes(image, 0, 2)
    image_rot = np.swapaxes(tmp_rot, 0, 1)

    return image_rot


# Function for reducing the frame per seconds
def change_fps_down(input_video_path, fps):
    video_array = load_video_as_array(input_video_path)
    video = cv2.VideoCapture(input_video_path)
    fps_current = int(video.get(cv2.CAP_PROP_FPS))
    height, width, frame_count = video_array.shape

    # Finding the ratio of current fps and desired fps
    frame_skip = int(round(fps_current / fps))
    new_frame_count = int(frame_count / frame_skip)
    new_video_array = np.empty((height, width, new_frame_count), dtype=np.uint8)

    # Skipping the undesired frames and getting the desired ones based on frame_skip ratio
    for i in range(new_frame_count):
        new_frame_idx = i * frame_skip

        new_video_array[:, :, i] = video_array[:, :, new_frame_idx]

    return new_video_array


# Function for increasing the frame per seconds
def change_fps_up(input_video_path, fps):
    video_array = load_video_as_array(input_video_path)
    print('up', video_array.shape)
    video = cv2.VideoCapture(input_video_path)
    fps_current = int(video.get(cv2.CAP_PROP_FPS))
    height, width, frame_count = video_array.shape
    # Finding the ratio of current fps and desired fps
    frame_skip = (fps_current / fps)

    new_frame_count = int(frame_count / frame_skip)

    new_video_array = np.empty((height, width, new_frame_count), dtype=np.uint8)

    # Interpolating new frames according to frame_skip ratio by weighted interpolation
    for i in range(new_frame_count):
        new_frame_idx = int(i * frame_skip)

        if new_frame_idx == 0:
            new_video_array[:, :, i] = video_array[:, :, new_frame_idx]
        else:
            prev_frame_idx = int((i - 1) * frame_skip)
            next_frame_idx = new_frame_idx
            prev_frame = video_array[:, :, prev_frame_idx]
            next_frame = video_array[:, :, next_frame_idx]
            interpolated_frame = cv2.addWeighted(prev_frame, 0.5, next_frame, 0.5, 0)
            new_video_array[:, :, i] = interpolated_frame.astype(np.uint8)

    return new_video_array


def black_object_detection(image, low_threshold=100, high_threshold=190, kernel_size=45):
    # Canny edge detection is applied to find the black object in front of the valve
    edges = cv2.Canny(image, low_threshold, high_threshold)

    # Generating the mask for thresholding
    edges_hysterosis = np.zeros_like(edges, dtype=np.uint8)

    strong_edges = edges > high_threshold
    edges_hysterosis[strong_edges] = 255

    while True:
        weak_edges = edges > low_threshold
        weak_edges = np.logical_and(weak_edges, ~strong_edges)

        if not np.any(weak_edges):
            break

        strong_edges = np.logical_or(strong_edges, weak_edges)
        edges_hysterosis[weak_edges] = 255

    # Perform morphological operations to include pixels inside the edges
    kernel = np.ones((7, 7), np.uint8)
    edges_hysterosis = cv2.dilate(edges_hysterosis, kernel, iterations=1)

    return edges_hysterosis


def replace_stitched_pixels(image, binary_mask, kernel_size=30):
    # Iterating over each pixel in the image
    for i in range(image.shape[0]):
        print('done1')
        for j in range(image.shape[1]):
            print('done2')
            if binary_mask[i, j] > 0:  # Check if the pixel is marked as stitched
                print('Stitched pixel found at ({}, {})'.format(i, j))

                # Extract the surrounding non-stitched pixels with kernel excluding white pixels
                surrounding_pixels = []
                for m in range(-kernel_size // 2, kernel_size // 2 + 1):
                    for n in range(-kernel_size // 2, kernel_size // 2 + 1):
                        if (0 <= i + m < image.shape[0]) and (0 <= j + n < image.shape[1]) and binary_mask[
                            i + m, j + n] != 255:
                            surrounding_pixels.append(image[i + m, j + n])

                # Calculate the mean gray value of the surrounding non-stitched pixels
                if len(surrounding_pixels) == 0:
                    image[i, j] = 150
                else:
                    mean_gray_value = np.mean(surrounding_pixels)

                    # Replace the gray value of the stitched pixel with the mean gray value
                    image[i, j] = int(mean_gray_value)

    return image


def segment_valve_opening(image, frame_boundry, referance_image):
    first_frame_gray = image[:, :, 0]
    # Taking 5 frame after the contrast start frame to see the whole valve
    middle_frame_index = frame_boundry + 5
    middle_frame_gray = image[:, :, middle_frame_index]
    differentiated_pixels = cv2.absdiff(first_frame_gray, middle_frame_gray)

    # To find the mostly changed pixels the threshold is set
    diff_threshold = 30
    # Binary mask has created to encapsulate mostly changed pixels
    binary_mask_diff = (differentiated_pixels > diff_threshold).astype(np.uint8) * 255
    # Kernels has set for morphological operations
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Morphological operations applied for clearing the smaller contrasts
    opening_op_binary = cv2.morphologyEx(binary_mask_diff, cv2.MORPH_OPEN, kernel1)
    closing_op_binary = cv2.morphologyEx(opening_op_binary, cv2.MORPH_CLOSE, kernel2)

    # Extracting the objects found in operation
    changed_contours, _ = cv2.findContours(closing_op_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Largest contour indicating the valve area has extracted
    largest_valve_contour = max(changed_contours, key=cv2.contourArea)

    # Finding the coordinates of the valve area by creating a rectangle
    x_axis, y_axis, width, height = cv2.boundingRect(largest_valve_contour)

    # Getting the center coordinates of the rectangle
    center_x = x_axis + (width // 2)
    center_y = y_axis + (height // 2)

    # Setting the sizes to get (500,500) box
    half_size = 250
    new_width = new_height = 500
    # Based on the found coordinates apply some general shifts to reach the opening region (Decided based on observations)
    if center_x <= 200:
        shift_x = 150
    else:
        shift_x = 50

    if center_y <= 420:
        shift_y = 0
    else:
        shift_y = -70

    # Making sure the dimentions of bounding box is not exceeding the edges
    new_x_axes = max(0, (center_x - half_size + shift_x))
    new_y_axes = max(0, (center_y - half_size + shift_y))

    new_y_axes = min(524, new_y_axes)
    new_x_axes = min(524, new_x_axes)

    segmented_middle_frame = middle_frame_gray[new_y_axes:new_y_axes + new_height, new_x_axes:new_x_axes + new_width]

    # Applying Chi-squared shift based on referance image
    noise = 0.01
    xoff, yoff, exoff, eyoff = chi2_shift(referance_image, segmented_middle_frame, noise, return_error=True,
                                          upsample_factor='auto')

    # Adjusting new_x_axes and new_y_axes based on the calculated offsets and checking the edges
    new_y_axes = max(0, int(round(new_y_axes - yoff)))
    new_x_axes = max(0, int(round(new_x_axes - xoff)))

    new_y_axes = min(524, new_y_axes)
    new_x_axes = min(524, new_x_axes)

    return new_x_axes, new_y_axes, new_height, new_width


def process_videos_in_dir(video_dir, output_dir):
    # Getting a list of all video files in the directory
    video_files = glob.glob(os.path.join(video_dir, '*.avi'))
    output_videos = output_dir
    # Looping through each video file and process it
    for video_file in video_files:
        # Calling the load_video_as_array function with the current video file
        # Checking the fps
        standard_fps = 8
        video = cv2.VideoCapture(video_file)

        fps = int(video.get(cv2.CAP_PROP_FPS))
        print(fps)
        if fps == standard_fps:
            image = load_video_as_array(video_file)
            print(image.shape)
        elif fps > standard_fps:
            image = change_fps_down(video_file, standard_fps)
            print(image.shape)
        elif fps < standard_fps:
            image = change_fps_up(video_file, standard_fps)
            print(image.shape)

        kernel = np.ones((15, 15), np.uint8)
        # Counting the number of labels of each frames
        label_numbers = []
        for i in range(0, (image.shape[-1]) - 1):

            image_to_filter = image[:, :, i]
            # Cropping pulse vision from the image
            cropped_image = image_to_filter[0:(image.shape[1] - 100), 0:(image.shape[1])]
            # Standardization of image
            standardize = (cropped_image - cropped_image.mean()) / np.sqrt(cropped_image.var() + 1e-5)
            standardize -= standardize.min()
            standardize /= standardize.max()
            standardize *= 255  # [0, 255] range

            if standardize.min() <= 0:
                standardize = standardize - (standardize.min())

            # Adding Gaussian blur to image
            blured_image = cv2.GaussianBlur(standardize, (9, 9), 0)
            blured_image_8bit = cv2.convertScaleAbs(blured_image)

            #  Applying Otsu Thresholding
            ret, thresh1 = cv2.threshold(blured_image_8bit, 120, 255, cv2.THRESH_BINARY +
                                         cv2.THRESH_OTSU)
            # Applying Morphological operations
            opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            invert_binary_img = cv2.bitwise_not(closing)
            num_labels, labels = cv2.connectedComponents(invert_binary_img)
            label_hue = np.uint8(179 * labels / np.max(labels))
            blank_ch = 255 * np.ones_like(label_hue)
            labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
            labeled_img_converted = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
            labeled_img_converted[label_hue == 0] = 0

            # Counting the labels calculated from the algorithm above
            label_numbers.append(num_labels)

        # Finding valleys on the graph
        kernel_size2 = 1
        kernel2 = np.ones(kernel_size2) / kernel_size2
        label_data_convolved = np.convolve(label_numbers, kernel2, mode='same')

        # Adding Gauss noise to the valley graph
        seed = 500
        np.random.seed(seed)
        noise = np.random.normal(0.0, 0.05, size=label_data_convolved.shape)
        convolved_noisy = label_data_convolved + noise

        # Finding the peaks of the label numbers
        peaks2 = argrelextrema(convolved_noisy, np.less)

        # Creating the boundry based on the lenght of the video
        image_sorted = sorted(image.shape)
        # Selecting the correct local minimum based on the length of the video (Decided based on observations)
        if image_sorted[0] < 25:
            frame_boundry = peaks2[0][1]
        elif 25 <= image_sorted[0] <= 50:
            frame_boundry = peaks2[0][2]
        elif image_sorted[0] > 50:
            frame_boundry = peaks2[0][3]

        # Stitch removal applied if necessary
        # for i in range(0, (image.shape[-1]) - 1):
        # image_to_remove = image[:, :, i]
        # stitches_on_image = black_object_detection(image_to_remove)
        # removed_image = replace_stitched_pixels(image_to_remove, stitches_on_image)
        # image[:, :, i] = removed_image

        # Applying the segmentation function
        new_x, new_y, new_h, new_w = segment_valve_opening(image, frame_boundry, reference_image)
        print(new_x, new_y, new_h, new_w)

        # Extracting important frames and segmented area
        for i in range(frame_boundry,
                       frame_boundry + 15):  # changed recently, -3 error is neglected to be sure that including openning
            # Load an image in the greyscale
            image_to_filter = image[:, :, i]
            # Cropping the image
            cropped_image = image_to_filter[0:(image.shape[1] - 100), 0:(image.shape[1])]
            # Standardization of image
            standardize = (cropped_image - cropped_image.mean()) / np.sqrt(cropped_image.var() + 1e-5)
            standardize -= standardize.min()
            standardize /= standardize.max()
            standardize *= 255  # [0, 255] range

            if standardize.min() <= 0:
                standardize = standardize - (standardize.min())

            # Adding Gaussian blur to image
            blured_image = blur = cv2.GaussianBlur(standardize, (9, 9), 0)
            blured_image = blured_image[new_y:new_y + new_h, new_x:new_x + new_w]
            blured_image_8bit = cv2.convertScaleAbs(blured_image)

            cv2.imwrite(r'Important_frames\%s.jpg' % i, blured_image_8bit)

        frameSize = (500, 500)  # It should be checked for each video since size of the all videos are not the same
        video_name = video_file.split("\\")[1]

        video_name, video_ext = os.path.splitext(video_name)

        out = cv2.VideoWriter(f'{output_videos}\%s.avi' % video_name, cv2.VideoWriter_fourcc(*'DIVX'), standard_fps,
                              frameSize)

        for filename in sorted(glob.glob('Important_frames/*.jpg'), key=len):
            img = cv2.imread(filename)
            out.write(img)

        # Remove all files from Important_frames directory
        for filename in os.listdir('Important_frames'):
            os.remove(os.path.join('Important_frames', filename))


if __name__ == '__main__':
    run()
