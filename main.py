import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui

def process_img(printscreen):
     processed_img = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
     processed_img = cv2.Canny(processed_img, threshold1=100, threshold2=300)
     return processed_img

def main(): 
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    
    last_time = time.time()
    while(True):
        # 800x600 windowed mode for GTA 5, at the top left position of your main screen.
        # 40 px accounts for title bar. 
        printscreen =  np.array(ImageGrab.grab(bbox=(0,40,765,534)))
        new_screen = process_img(printscreen)

        m_x, m_y = pyautogui.position()
        print(m_x,m_y)

        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        cv2.imshow('window', new_screen)
        # cv2.imshow('window2',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()
