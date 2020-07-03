# coding: utf-8

# # Advanced Lane Finding Using OpenCV
# **In this project, I used OpenCV to write a software pipeline to identify the lane boundaries in a video from a front-facing camera on a car.**

# ## Pipeline architecture:
# - **Compute Camera Calibration.**
# - **Apply Distortion Correction**.
# - **Apply a Perspective Transform.**
# - **Create a Thresholded Binary Image.**
# - **Define the Image Processing Pipeline.**
# - **Detect Lane Lines.**
# - **Determine the Curvature of the Lane and Vehicle Position.**
# - **Visual display of the Lane Boundaries and Numerical Estimation of Lane Curvature and Vehicle Position.**
# - **Process Project Videos.**
# 
# I'll explain each step in details below.

# #### Environement:
# -  Ubuntu 16.04
# -  Anaconda 5.0.1
# -  Python 3.6.2
# -  OpenCV 3.1.0

# In[1]:


# Importing Python libraries
import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from ipywidgets import interact, interactive, fixed
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[2]:


def display(img1, img2, lbl1, lbl2, x, y, img3=[], lbl3=[], cmap=None, n = 2):
    """
    Diplay the input images side-by-side.
        Parameters:
            img1: Input image #1.
            img2: Input image #2.
            lbl1: Label for input image #1.
            lbl2: Label for input image #2.
            x, y: Figure size.
            cmap (Default = None): Used to display gray images.
    """
    plt.figure(figsize=(x, y))
    plt.subplot(1, n, 1)
    plt.imshow(img1, cmap = cmap)
    plt.xlabel(lbl1, fontsize=15)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, n, 2)
    plt.imshow(img2, cmap = cmap)
    plt.xlabel(lbl2, fontsize=15)
    plt.xticks([])
    plt.yticks([])
    if n == 3:
        plt.subplot(1, n, 3)
        plt.imshow(img3, cmap = cmap)
        plt.xlabel(lbl3, fontsize=15)
        plt.xticks([])
        plt.yticks([])
    plt.show()


# ---
# ## Step 1: Compute Camera Calibration

# The OpenCV functions `cv2.findChessboardCorners()` and `cv2.drawChessboardCorners()` are used for image calibration. We have 20 images of a chessboard, located in `./camera_cal`, taken from different angles with the same camera, and we'll use them as input for camera calibration routine.
# 
# `cv2.findChessboardCorners()` attempts to determine whether the input image is a view of the chessboard pattern and locate the internal chessboard corners, and then `cv2.drawChessboardCorners()` draws individual chessboard corners detected.
# 
# Arrays of object points, corresponding to the location of internal corners of a chessboard, and image points, the pixel locations of the internal chessboard corners determined by `cv2.findChessboardCorners()`, are fed to `cv2.drawChessboardCorners()` which returns camera calibration and distortion coefficients.
# 
# 
# These will then be used by the OpenCV `cv2.calibrateCamera()` to find the camera intrinsic and extrinsic parameters from several views of a calibration pattern. These parameters will be fed to `cv2.undistort` function to correct for distortion on any image produced by the same camera.

# In[5]:


cal_images = glob.glob('camera_cal/*.jpg')
test_images = glob.glob('test_images/*.jpg')

nx, ny = 9, 6
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)


# In[6]:


def calibrate_camera(cal_images, nx, ny):
    """
    Compute camera calibration and return the camera intrinsic and extrinsic parameters.
        Parameters:
            cal_images: A list of the chessboard calibration images.
            nx, ny: Chessboard dimensions.
    """
    objpoints = []  # 3D points
    imgpoints = []  # 2D points
    
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)

    for file in cal_images:
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    return mtx, dist


# In[7]:


mtx, dist = calibrate_camera(cal_images, nx, ny)


# ---
# ## Step 2: Apply Distortion Correction

# OpenCV provides `cv2.undistort` function, which transforms an image to compensate for radial and tangential lens distortion.

# In[8]:


def undistort(img, mtx, dist):
    """
    Use the camera calibration parameters to correct the input image for distortion.
        Parameters:
            img: Input image.
            mtx: Output floating-point camera matrix.
            dist: Output vector of distortion coefficients.
    """
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


# In[9]:



# The effect of `undistort` is particularly noticeable, by the change in shape of the car hood at the bottom corners of the image.

# ---
# ## Step 3: Apply a Perspective Transform

# A common task in autonomous driving is to convert the vehicle’s camera view of the scene into a top-down “bird’s-eye” view. We'll use OpenCV's `cv2.getPerspectiveTransform()` and `cv2.getPerspectiveTransform()` to do this task.

# In[11]:


# In[12]:


# Define the region of interest
src = np.float32([[190, 700], [1110, 700], [720, 470], [570, 470]])

bottom_left = src[0][0]+100, src[0][1]
bottom_right = src[1][0]-200, src[1][1]
top_left = src[3][0]-250, 1
top_right = src[2][0]+200, 1
dst = np.float32([bottom_left, bottom_right, top_right, top_left])


# In[13]:


def perspective_transform(img, src, dst):
    """
     Convert the vehicle’s camera view of the scene into a top-down “bird’s-eye” view.
        Parameters:
            img: Input image.
            src: Source points.
            dst: Destination points.
    """
    image_shape = img.shape
    img_size = (image_shape[1], image_shape[0])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    # Return the resulting image and matrix
    return warped, M, Minv


# In[14]:


# Applying perspective transform to several test_images

# In[15]:





# ---
# ## Step 4: Create a Thresholded Binary Image

# Now, we will use color transform and Sobel differentiation to detect the lane lines in the image.

# ### Exploring different color spaces

# #### RGB color space:

# In[16]:



# #### HSV color space:
# This type of color model closely emulates models of human color perception. While in other color models, such as RGB, an image is treated as an additive result of three base colors, the three channels of HSV represent hue (H gives a measure of the spectral composition of a color), saturation (S gives the proportion of pure light of the dominant wavelength, which indicates how far a color is from a gray of equal brightness), and value (V gives the brightness relative to
# the brightness of a similarly illuminated white color) corresponding to the intuitive appeal of tint, shade, and tone.

# In[17]:



# #### LAB color space:
# The Lab color space describes mathematically all perceivable colors in the three dimensions L for lightness and a and b for the color opponents green–red and blue–yellow.

# In[18]:



# #### HLS color space:
# This model was developed to specify the values of hue, lightness, and saturation of a color in each channel. The difference with respect to the HSV color model is that the lightness of a pure color defined by HLS is equal to the lightness of a medium gray, while the brightness of a pure color defined by HSV is equal to the brightness of white.

# In[19]:



# ### Color Space Thresholding

# As you may observe, the white lane lines are clearly highlighted in the L-channel of the of the HLS color space, and the yellow line are clear in the L-channel of the LAP color space as well. We'll apply HLS L-threshold and LAB B-threshold to the image to highlight the lane lines.

# In[20]:


def hls_l_thresh(img, thresh=(220, 255)):
    """
     Threshold the input image to the L-channel of the HLS color space.
        Parameters:
            img: HLS image.
            thresh: Minimum and Maximum color intensity.
    """
    img = img[:,:,1]
    img = img*(255/np.max(img))
    binary_output = np.zeros_like(img)
    binary_output[(img > thresh[0]) & (img <= thresh[1])] = 1
    return binary_output


# In[21]:


# In[22]:


def lab_b_thresh(img, thresh=(190, 255)):
    """
     Threshold the input image to the B-channel of the LAB color space.
        Parameters:
            img: LAB image.
            thresh: Minimum and Maximum color intensity.
    """
    img = img[:,:,2]
    if np.max(img) > 175:
        img = img*(255/np.max(img))
    binary_output = np.zeros_like(img)
    binary_output[(img > thresh[0]) & (img <= thresh[1])] = 1
    return binary_output


# In[23]:


# In[24]:


def threshold_color_space(img):
    """
     Threshold the input image to the L-channel of the HLS color space and the B-channel of the LAB color space.
        Parameters:
            img: Input image.
    """
    img_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    img_LAB = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    img_thresh_HLS = hls_l_thresh(img_HLS)
    img_thresh_LAB = lab_b_thresh(img_LAB)
    combined_img = np.zeros_like(img_thresh_HLS)
    combined_img[((img_thresh_HLS == 1) | (img_thresh_LAB == 1))] = 1
    return combined_img


# In[25]:


# ### Sobel Differentiation

# Now, we'll explore different Sobel differentiation techniques, and try to come up with a combination that produces a better output than color space thresholding.

# In[26]:


def abs_sobel(img, orient='x', sobel_kernel=3, thresh=(25, 255)):
    """
    Apply absolute Sobel diffrentiation to the input image.
        Parameters:
            img: Input image.
            orient (Default = x): Gradients direction.
            sobel_kernel (Default = 3): Size of the extended Sobel kernel.
            thresh (Default = (25, 255)): Minimum and Maximum gradient strength.
    """
    sobel = cv2.Sobel(img, cv2.CV_64F, orient=='x', orient=='y', ksize= sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sxbinary


# In[27]:



# In[28]:




# In[29]:


def mag_sobel(img, sobel_kernel=15, thresh=(25, 255)):
    """
    Apply magnitude Sobel diffrentiation to the input image.
        Parameters:
            img: Input image.
            sobel_kernel (Default = 15): Size of the extended Sobel kernel.
            thresh (Default = (25, 255)): Minimum and Maximum gradient strength.
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    mag_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sxbinary


# In[30]:


# In[31]:


# In[32]:


def dir_sobel(img, sobel_kernel=25, thresh=(0, 0.09)):    
    """
    Apply direction Sobel diffrentiation to the input image.
        Parameters:
            img: Input image.
            sobel_kernel (Default = 25): Size of the extended Sobel kernel.
            thresh (Default = (0, 0.09)): Minimum and Maximum gradient strength.
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel) 
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel) 
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    binary_output =  np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    return binary_output


# In[33]:



# In[35]:


def combined_sobel(img, mag_kernel_size, mag_thresh):
    """
    Apply both absolute and magnitude Sobel diffrentiation to the input image.
        Parameters:
            img: Input image.
            mag_kernel_size: Size of the extended Sobel kernel.
            mag_thresh: Minimum and Maximum gradient strength.
    """
    img_abs = abs_sobel(img)
    img_mag = mag_sobel(img, mag_kernel_size, mag_thresh)
    combined_img = np.zeros_like(img_mag)
    combined_img[((img_abs == 1) & (img_mag == 1))] = 1
    return combined_img


# In[36]:




# Now, we'll combine the absolute+magnitude Sobel outputs of both HLS and LAB.

# In[38]:


def combined_sobel_colors(img, mag_kernel_size, mag_thresh):
    """
    Combine Sobel diffrentiation results from applying to diffrenet color spaces.
        Parameters:
            img: Input image.
            mag_kernel_size: Size of the extended Sobel kernel.
            mag_thresh: Minimum and Maximum gradient strength.
    """
    img_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    img_LAB = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    combined_HLS_L = combined_sobel(img_HLS[:,:,1], mag_kernel_size, mag_thresh)
    combined_LAB_B = combined_sobel(img_LAB[:,:,2], mag_kernel_size, mag_thresh)
    combined_sobel_colors_img = np.zeros_like(combined_HLS_L)
    combined_sobel_colors_img[((combined_HLS_L == 1) | (combined_LAB_B == 1))] = 1
    return combined_sobel_colors_img


# In[39]:



# ### Comparison between Color Thresholding and Sobel Diffrentiation

# We'll apply both color thresholding and Sobel diffrentiation to all the test images to explore which of these two techniques will be better to do the task.

# In[40]:

canny= []
color_threshold = []
sobel_diff = []
test_images_warped = []
for file in test_images:
    image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
    image = undistort(image, mtx, dist)
    image = perspective_transform(image, src, dst)[0]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)
    image= cv2.cvtColor(cl1,cv2.COLOR_GRAY2RGB)
    canny_img= cv2.Canny(image, 190, 230)
    canny.append(canny_img)
    test_images_warped.append(image)
    color_threshold.append(threshold_color_space(image))
    sobel_diff.append(combined_sobel_colors(image, 15, (25, 225)))


# In[41]:


for orginal, color, sobel, canny_img in zip( test_images_warped, color_threshold, sobel_diff, canny):
    display(color,
            sobel,
            'color image',
            'sobelimage',
            14,
            7,
            canny_img,
            'canny image',
            cmap='gray',
            n = 3)
    #new=  cv2.bitwise_and(color,sobel )
    #new=cv2.bitwise_or((color, sobel ,dst=None,mask=None )
    canny_img = cv2.Canny(orginal, 50, 230)
    new=cv2.addWeighted( color, 1, sobel, 1, 0 ,dst=None ,dtype=cv2.CV_64F)
    new=cv2.addWeighted(  new, 1, canny_img , 1, 0 ,dst=None ,dtype=cv2.CV_64F)

    kernel = np.ones((1,1),np.uint8)
    new= cv2.morphologyEx(new, cv2.MORPH_OPEN, kernel)
    display(new,
            color,
            'weighted image',
            'Color thresholded image',
            14,
            7,
            sobel,
            'Sobel diffrentiated image',
            cmap='gray',
            n = 3)
    
    
def image_process(img):
  """
    # Undistort
    img = undistort(img, mtx, dist)
    # Perspective Transform
    img, M, Minv = perspective_transform(img, src, dst)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #blur=cv2.GaussianBlur(gray, (5, 5), 0)
    # input to CLAHE
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)
    image = cv2.cvtColor(cl1, cv2.COLOR_GRAY2RGB)
    combined_sobel_colors_img = combined_sobel_colors(image, 15, (25, 225))
    #This section for sobel
    #transform from grey to CM
    #colormap=cv2.applyColorMap(cl1, cv2.COLORMAP_JET)
    #combined sobal on the CM
    sobel = combined_sobel(cl1, 15, (25, 225))*255
    #making binary image of result
    #grayImage = cv2.cvtColor(sobel, cv2.COLOR_BGR2GRAY) #changed sobal colormap to binary
    #(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    #reducing noise 
    #kernel = np.ones((2,2),np.uint8)

    #lastfinal= cv2.add(final ,opening_combined_sobel_colors,dst=None,mask=None,dtype=cv2.CV_64F)
    #final =cv2.addWeighted(  sobel, 0.8, combined_sobel_colors_img , 1, 0 ,dst=None ,dtype=cv2.CV_64F)
    final = cv2.bitwise_or(sobel, combined_sobel_colors_img)
    #lastfinal= cv2.addWeighted(   opening_combined_sobel_colors, 0.8,final , 1, 0 ,dst=None ,dtype=cv2.CV_64F)
    #final= cv2.add(opening_combined_sobel,opening_combined_sobel_colors,dst=None,mask=None,dtype=cv2.CV_64F)
    kernel2= np.ones((2,2),np.uint8)
    final= cv2.morphologyEx(final, cv2.MORPH_OPEN, kernel2)
    display(final, img, 'lastfinal', 'prespective', 14, 7, cmap='gray')
    return final, Minv

test_images_imgs = []
test_images_processed = []
for file in test_images:
    image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
    test_images_imgs.append(image)
    image, Minv = image_process(image)
    test_images_processed.append(image)


# In[44]:


for original, processed in zip(test_images_imgs, test_images_processed):
    display(original,
            processed,
            'Original test image',
            'Processed test image',
            14,
            7,
            cmap='gray')
"""