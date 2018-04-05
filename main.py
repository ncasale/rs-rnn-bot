from __future__ import division
import numpy as np #pip install numpy
from PIL import ImageGrab # pip3 install Pillow
import cv2 #pip install opencv-python
import time
import pyautogui # pip install pyautogui
import win32api #pip install pywin32
from matplotlib import pyplot as plt #pip install matplotlib
from math import cos, sin

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

def process_img(printscreen): #take the BGR image and convert it to greyscale to use for edge detection processing
     processed_img = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
     processed_img = cv2.Canny(processed_img, threshold1=100, threshold2=300)
     return processed_img

def main(): 
    for i in list(range(4))[::-1]:    #Provides a 4s countdown before the recording starts to allow time to click into the window.
        print(i+1)
        time.sleep(1)
    
    last_time = time.time()
    while(True):
        #Set up array to collect screen data with the dimensions listed below. 40px offset to account for navigation bar
        printscreen =  np.array(ImageGrab.grab(bbox=(0,40,765,534)))
        new_screen = process_img(printscreen)

        #collect current mouse location and print it to the console (there are better ways of doing this with pywin32 that we should look into to allow for recording of mouse clicks)
        m_x, m_y = pyautogui.position()
        print(m_x,m_y)

        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()

        #Top line shows the edge detection window. Line below shows color image
        # cv2.imshow('window', new_screen)
        feed = cv2.imshow('window2',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()