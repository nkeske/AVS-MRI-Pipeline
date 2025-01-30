
# Aortic Valve Stenosis - Preprocessing

This program applies various preprocessing methods to find frames of interests in the input videos, subsequently crop and segment the heart valve opening area in the corresponding frames. 

## Requirements
### Libraries
Python 3.x\
OpenCV (version 4.5.3)\
NumPy (version 1.21.1)\
Matplotlib (version 3.4.3)\
Panel (version 0.12.1)\
Scipy (version 1.7.1)\
Scikit-image (version 0.18.3)\
Image_registration (version 1.3.0)

### Directory Structure
- `AVS_Preprocessing.py`
- `Input_videos/`
    - `video1.avi`
    - `video2.avi`
    - ...
- `Output_videos/`
- `Important_frames/`
- `reference_image.jpg`







## How to Run

1. Install the necessary libraries using the following command:

    ```bash
    pip install opencv-python==4.5.3 numpy==1.21.1 matplotlib==3.4.3 panel==0.12.1 scipy==1.7.1 scikit-image==0.18.3 image-registration==1.3.0
    ```

2. Make sure you have the video that will be preprocess are in the "Input_videos" directory and a reference image, "reference_image.jpg", 'Important_frames' as well as 'Output_videos' are in the same directory as the script.

3. Run the script by executing the following command:

    ```bash
    python AVS_Preprocessing.py
    ```

4. The processed videos will be saved in the "Output_videos" directory in the end of preprocessing.
#

- Author: Nazligul Keske
- Supervisor: M.Sc. Annika Engel
  