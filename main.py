"""Login for the account:
U/N: sith6522@gmail.com
P/W: opencvftw
"""

from __future__ import division
import numpy as np #pip install numpy
from PIL import ImageGrab # pip3 install Pillow
import cv2 #pip install opencv-python
import time
import pyautogui # pip install pyautogui
from matplotlib import pyplot as plt #pip install matplotlib
from math import cos, sin
import os
import sys
import datetime

"""
def detect(feed,image):

    #Scale the image to be 700x700
    max_dimension = max(image.shape)
    scale = 700/max_dimension
    image = cv2.resize(image, none, fx=scale, fy=scale)

    #Clean the image 
    image_blur = cv2.GaussianBlur(image, (7,7), 0)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    #Define filters to filter by color
    min_color = np.array([])
"""

def process_img(printscreen):
    """Take BGR formatted image and convert to greyscale. Used for edge detection processing.
    
    Returns:
        numpy array -- A greyscale image run through a Canny function
    """
     processed_img = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
     processed_img = cv2.Canny(processed_img, threshold1=100, threshold2=300)
     return processed_img


def writeImageToFile(image, current_frame, path, frame_rate):
    """Write the passed image to a file
    
    Arguments:
        image {numpy array} -- the image to write to file
        current_frame {int} -- current frame
        path {string} -- path of write file
        frame_rate {int} -- frame rate of application
    """

    #Only write to file once per second
    if current_frame % frame_rate == 0:
        file_name = path + '/image_detection_training_' + str(int(current_frame/frame_rate)) + '.png'
        cv2.imwrite(file_name, image)

def createTrainingDataDir():
    """Creates the directory housing training images if it doesn't exist. The directories are 
       created in the format: YYYYMMDDHH.   
    
    Returns:
        string -- the path to the training directory
    """
    #Get current datetime
    now = datetime.datetime.now()

    #Format day/month/hour correctly
    month = str(now.month) if now.month >= 10 else '0' + str(now.month)
    day = str(now.day) if now.day >= 10 else '0' + str(now.day)
    hour = str(now.hour) if now.hour >= 10 else '0' + str(now.hour)

    #Create path to directory
    date_string = str(now.year) + month + day + hour
    path = './training_images/' + date_string
    if not os.path.exists(path):
        print('No directory found. Making new directory at ' + date_string)
        os.makedirs(path)
    else:
        print('Directory found!')
    return path

def labelimg():
    """
    1: /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
       git clone https://github.com/tzutalin/labelImg.git
    2: brew install qt  # will install qt-5.x.x
    3: brew install libxml2
    4: navigate to the labelImg directory
    5: make qt5py3 
    6: python3 labelImg.py
    """
    
def main():
    #Provide 4 second countdown before recording to navigate to window 
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    
    #Time used for loop timing
    last_time = time.time()

    #Enter the frame rate you get when you run this
    frame_rate = 5

    #Current frame 
    current_frame = 0

    #Create data directory for this run
    training_data_path = createTrainingDataDir()

    while(True):
        #Increment frame count
        current_frame += 1

        #Set up array to collect screen data with the dimensions listed below. 40px offset to account for navigation bar
        image_x = 960
        image_y = 660
        printscreen =  np.array(ImageGrab.grab(bbox=(0,40,image_x, image_y)))
        new_screen = process_img(printscreen)

        #collect current mouse location and print it to the console (there are better ways of doing this with pywin32 that we should look into to allow for recording of mouse clicks)
        m_x, m_y = pyautogui.position()
        print(m_x,m_y)

        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()

        #Top line shows the edge detection window. Line below shows color image
        # cv2.imshow('window', new_screen)
        image = cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB)

        #Save image to file
        writeImageToFile(image, current_frame, training_data_path, frame_rate)

        #Show the second window
        feed = cv2.imshow('window2',image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()