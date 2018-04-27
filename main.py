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
import pyHook #pip install pyHook-1.5.1-cp36-cp36m-win_amd64.whl
import pythoncom # pip install pywin32 -- WINDOWS ONLY


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
def left_down(event):
    print ("left click")
    print ('Position:',event.Position)
    return True  
                

def right_down(event):
    print ("right click")
    print ('Position:',event.Position)
    return True    
    
def OnKeyboardEvent(event):
    print (event.Key)
    return True    
    
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
    image_count, training_data_path = createTrainingDataDir()
    current_image = int(image_count)
    current_frame = int(current_image*frame_rate)


    while(True):
        #Increment frame count
        current_frame += 1

        #Set up array to collect screen data with the dimensions listed below. 40px offset to account for navigation bar
        image_x = 960
        image_y = 660
        printscreen =  np.array(ImageGrab.grab(bbox=(0,40,image_x, image_y)))
        new_screen = process_img(printscreen)

        #collect current mouse location and print it to the console (there are better ways of doing this with pywin32 that we should look into to allow for recording of mouse clicks)
        # m_x, m_y = pyautogui.position()
        hm.HookMouse()
        hm.HookKeyboard()

        while(left_down == True):  ## Todo: Right now, this works for registering when a mouse button is clicked but cant register when a key/mouse button is lifted
            pythoncom.PumpMessages()


        #check how long the loop took to run
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
            hm.UnhookMouse()
            hm.UnhookKeyboard()
            break

def writeImageToFile(image, frame_count, path, frame_rate):
    #Create a new directory for this run of images
    if frame_count % frame_rate == 0:
        file_name = path + '/image_detection_training_' + str(int(frame_count/frame_rate)) + '.png'
        cv2.imwrite(file_name, image)

def createTrainingDataDir():
    #Create a new directory for this run of images
    now = datetime.datetime.now()
    #TODO: datetime formatter instead of concat
    date_string = str(now.year) + str(now.month) + str(now.day) + str(now.hour)
    path = './training_images/' + date_string
    frame_count = 0 

    #Check if a directory exists for this hour. If there is, open the directory and save images to that. If not, create a new directory.
    if not os.path.exists(path):
        print('No directory found. Making new directory at ' + date_string)
        os.makedirs(path)
        file_name = 0
    else:
        print('Directory found! Checking current number.')         
        if not os.listdir(path) == []:
            files = os.listdir(path)
            full_list = [os.path.join(path,i) for i in files]
            time_sorted_list = sorted(full_list, key=os.path.getmtime)
            file_name = time_sorted_list[-1]
            file_name = file_name.strip(path)
            file_name = file_name.strip('\image_detection_training_')
            file_name = file_name.strip('.png')
            print('Starting frame count: ' + file_name)
        else: 
            print('Folder empty. Starting frame count: 1')
            file_name = 0 
            #Current frame           
    return (file_name, path)

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
    


print("Starting Script...")
# hook mouse
hm = pyHook.HookManager()
hm.SubscribeMouseLeftDown(left_down)
hm.SubscribeMouseRightDown(right_down)

print("Mouse Hooked. Hooking Keyboard.")
# hook keyboard
hm.KeyDown = OnKeyboardEvent # watch for all keyboard events


print("Keyboard Hooked. Starting main.")
main()
