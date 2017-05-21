import os
import sys
import shutil
import getopt

import cv2
import numpy as np
import pandas as pd
import imutils

# define HSV thresholds of interest for colour "red"
red_hue_lower_top = 30
red_hue_upper_bot = 140
red_satur_bot = 40
red_value_bot = 30 #~brightness
red_lower_bot = np.array([0,red_satur_bot,red_value_bot])
red_lower_top = np.array([red_hue_lower_top,255,255])
red_upper_bot = np.array([red_hue_upper_bot,red_satur_bot,red_value_bot])
red_upper_top = np.array([179,255,255])

blue_hue_top = 140
blue_hue_bot = 100
blue_satur_bot = 40
blue_value_bot = 50 #~brightness
blue_bot = np.array([blue_hue_bot,blue_satur_bot,blue_value_bot])
blue_top = np.array([blue_hue_top,255,255])

def get_filenames(in_dir):
    # return a list of file names in in_dir. Returns ["test_in.png"] if 
    # in_dir is None

    if in_dir is None:
        filenames = ["test_in.png"] # in working directory
    else:
        filelist = os.listdir(in_dir)
        filenames = []
        for f in filelist: # filelist[:] makes a copy of filelist.
            if f.endswith(".png"):
                filenames.append(f)

    return filenames

def empty_dir(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def load_image(filepath):
    # load image
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img is None:
        print("ERROR: filename = %s could not be loaded" % filename)
    return img

x_co = 0                                                                       
y_co = 0                                                                       
def on_mouse(event,x,y,flag,param):                                            
    # used to return mouse position in display_pointer_hsv()
    global x_co                                                                
    global y_co                                                                
    if imutils.is_cv2():
        if(event==cv2.cv.CV_EVENT_MOUSEMOVE):                                          
            x_co=x                                                                 
            y_co=y                                                                 
    elif imutils.is_cv3():
        if(event==cv2.CV_EVENT_MOUSEMOVE):                                          
            x_co=x                                                                 
            y_co=y                                                                 

def display_pointer_hsv(window_name, img):
    # use this to insepct the HSV colour values of the original image
    # and set the HSV thresholds below
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:                                                                    
        dummy = img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)                                 
        if imutils.is_cv2():
            cv2.cv.SetMouseCallback(window_name, on_mouse, 0);                                 
            s=cv2.cv.Get2D(cv2.cv.fromarray(hsv), y_co, x_co)                                    
        elif imutils.is_cv3():
            cv2.SetMouseCallback(window_name, on_mouse, 0);                                 
            s=cv2.Get2D(cv2.fromarray(hsv), y_co, x_co)                                    
        text = "H: " + str(s[0])+ ", S: " + str(s[1]) + ", V: " + str(s[2])
        cv2.putText(dummy ,text ,(10,500), font, 4, (55,25,255),2)
        cv2.imshow(window_name, dummy) 
        if cv2.waitKey(10) == 27: # press esc to break loop
            break   

def equalise_value_local(bgr):
    #take BGR image and do local equalisation of the sat  histogram 
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,32))
    hsv[:,:,2] = clahe.apply(hsv[:,:,2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def equalise_saturation_local(bgr):
    #take BGR image and do local equalisation of the value histogram 
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,32))
    hsv[:,:,1] = clahe.apply(hsv[:,:,1])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def equalise_value(bgr):
    #take BGR image and equalise the value histogram in HSV space
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def shift_red(hue):
    # takes hue channel and shifts everything by -60 
    # blue will then be 60 and red will be 120
    #hue16 = hue.astype("uint16") # because adding 120 may go over 256
    #shift = (hue16 + 120) % 180
    #hue8 = hue.astype("uint8") 
    #return hue8
    return ( (hue.astype("uint16") + 120) % 180 )

def threshold_red(img, th):
    # take cv BGR image and return a red threshold image
    # th is a dictionary containing, for example
        #"hue_plus":30, #degrees relative to 0/180 degrees
        #"hue_minus":20,
        #"sat":40,
        #"val":50
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_limits_bottom = np.array([0, th["sat"], th["val"]])
    upper_limits_bottom = np.array([th["hue_plus"], 255, 255])
    lower_limits_top = np.array([180 - th["hue_minus"], th["sat"], th["val"]])
    upper_limits_top = np.array([179, 255, 255])
    red_bottom = cv2.inRange(hsv, lower_limits_bottom, upper_limits_bottom)
    red_top = cv2.inRange(hsv, lower_limits_top, upper_limits_top)
    return cv2.addWeighted(red_bottom, 1.0, red_top, 1.0, 0.0)

    # hsv[:,:,0] = shift_red(hsv[:,:,0]) #shift to that red hue = 120 degrees
    # lower_limits = np.array([120 - th["hue_minus"], th["sat"], th["val"]])
    # upper_limits = np.array([120 + th["hue_plus"], 255, 255])
    # return cv2.inRange(hsv, lower_limits, upper_limits)

def threshold_blue(img, th):
    # take cv BGR image and return a blue threshold image
    # th is a dictionary containing, for example
        #"hue_plus":20, #degrees relative to 120 degrees
        #"hue_minus":20,
        #"sat":40,
        #"val":50
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_limits = np.array([120 - th["hue_minus"], th["sat"], th["val"]])
    upper_limits = np.array([120 + th["hue_plus"], 255, 255])
    return cv2.inRange(hsv, lower_limits, upper_limits)

def remove_nonred(img):
    # take cv BGR image and remove parts that are not "red". 
    # returns only hue channel
    
    #convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ones and zeros whether saturation and value channels pass thresholds
    sat_val_thresh = cv2.inRange(hsv, red_lower_bot, np.array([180, 255, 255]))

    #deal with lower red section
    hue = np.multiply(hsv[:,:,0], (hsv[:,:,0] < red_hue_lower_top))
    red_lower = np.multiply(hue, sat_val_thresh)

    #deal with upper red section
    hue = np.multiply(hsv[:,:,0], (hsv[:,:,0] > red_hue_upper_bot) )
    red_upper = np.multiply(hue, sat_val_thresh)

    red = cv2.addWeighted(red_lower, 1.0, red_upper, 1.0, 0.0)

    return red

def get_red_image(img):
    #TODO: delete once threshold_red is used
    # take cv BGR image and return a red threshold image
    
    #convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only colors of interest
    red_lower = cv2.inRange(hsv, red_lower_bot, red_lower_top)
    red_upper = cv2.inRange(hsv, red_upper_bot, red_upper_top)
    red = cv2.addWeighted(red_lower, 1.0, red_upper, 1.0, 0.0)

    return red

def get_blue_image(img):
    #TODO: delete once threshold_blue is used
    # take cv BGR image and return  a blue threshold image
    
    #convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only colors of interest
    blue = cv2.inRange(hsv, blue_bot, blue_top)

    return blue

def get_circles(img, min_dist, min_r, max_r, p1, p2):

    # Try to find circles
    if imutils.is_cv2():
        circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT,1, min_dist, 
                                   param1=p1, param2=p2, minRadius=min_r, 
                                   maxRadius=max_r)
    elif imutils.is_cv3():
        circles = cv2.HoughCircles(img, cv2.CV_HOUGH_GRADIENT,1, min_dist, 
                                   param1=p1, param2=p2, minRadius=min_r, 
                                   maxRadius=max_r)
    if circles is not None:
        # drop unnecessary 3rd dim returned by HoughCircles
        circles = circles[0,:,:]
    return circles


def get_small_circles(img):
    min_dist = 1                                                               
    p1 = 50                                                                    
    p2 = 10                                                                    
    min_r = 20                                                                 
    max_r = 50                                                                 
    return  get_circles(img, min_dist, min_r, max_r, p1, p2)

def get_medium_circles(img):
    min_dist = 1                                                               
    p1 = 50                                                                    
    p2 = 15                                                                    
    min_r = 50                                                                 
    max_r = 100                                                                 
    return  get_circles(img, min_dist, min_r, max_r, p1, p2)

def get_big_circles(img):
    min_dist = 1                                                               
    p1 = 50                                                                    
    p2 = 25                                                                    
    min_r = 100                                                                 
    max_r = 175                                                                 
    return  get_circles(img, min_dist, min_r, max_r, p1, p2)

def get_small_medium_circles(img):
    circles_small = get_small_circles(img)
    circles_medium = get_medium_circles(img)
    if circles_small is not None and circles_medium is not None:
        return np.concatenate( (circles_small, circles_medium), axis=0)
    elif circles_small is not None:
        return circles_small
    elif circles_medium is not None:
        return circles_medium
    else:
        return None

def get_medium_big_circles(img):
    circles_medium = get_medium_circles(img)
    circles_big = get_big_circles(img)
    if circles_medium is not None and circles_big is not None:
        return np.concatenate( (circles_medium, circles_big), axis=0)
    elif circles_medium is not None:
        return circles_medium
    elif circles_big is not None:
        return circles_big
    else:
        return None

def is_overlap(c1, c2):
    #check if 2 circles overlap
    distance = ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5
    return distance < c1[2] + c2[2]

def remove_overlap(circles, accums):
    # only keep the strongest circle if there are overlaps
    return_cicles = []
    return_accums = []
    i_ordered = np.array(accums).argsort()
    for n, i in enumerate(i_ordered):
        c1 = circles[i]
        for j in i_ordered[n+1:]:
            c2 = circles[j]
            if is_overlap(c1, c2):
                break
        else:
            # if you get to the end of the first loop then add to return_cicles
            #return_cicles = np.vstack([return_cicles, c1])
            return_cicles.append(c1)
            return_accums.append(accums[i])

    return return_cicles, return_accums

def draw_circles(circles_, img):

    # Draw circles on img
    image = img.copy()
    if circles_ is not None:
        for i in circles_:
            # draw the outer circle
            cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(image,(i[0],i[1]),2,(0,0,255),3)
    return image

def display_image(img, name, display_hsv):

        # display image with circles drawn
        cv2.namedWindow(name)
        if display_hsv:
            display_pointer_hsv(name, img)
        else:
            cv2.imshow(name, img)
            cv2.waitKey(0)

def crop_ns_sign(img, circle):
    # crop img around a no-stopping sign defined by circle:[x, y, radius]
    x = circle[0]
    y = circle[1]
    r = circle[2]
    bb = np.array([y - 4*r, y + 10*r, x - 4*r, x + 4*r])
    for i, x in enumerate(bb):
        if x < 0:
            bb[i] = 0

    return img[bb[0]:bb[1], bb[2]:bb[3]] 

def pixels_in_circle(img, circle):
    # returns the number of pixels within a circle that are equale to 1
    # img: 2d array representing a threshholded image (elements = 0 or 1)
    # circle: array that looks like  [x, y, radius]

    height,width = img.shape
    circle_img = np.zeros((height,width), np.uint8)
    centre = (circle[0], circle[1])
    radius = circle[2]
    cv2.circle(circle_img, centre, radius, 1, thickness=-1)
    masked_data = cv2.bitwise_and(img, circle_img)

    return np.count_nonzero(masked_data)

def get_frac_blue_area(circle, circle_, blue_img):
    area  = np.pi * circle[2]**2
    blue_pix = pixels_in_circle(blue_img, circle_)
    return blue_pix / area

def get_features(circle, circle_, red_img, blue_img):
    area  = np.pi * circle[2]**2
    radius  = circle[2]
    red_pix = pixels_in_circle(red_img, circle_)
    blue_pix = pixels_in_circle(blue_img, circle_)
    x = circle[0]
    y = circle[1]

    dic = {}
    dic["red_pix"] = red_pix
    dic["blue_pix"] = blue_pix
    dic["frac_red_area"] = red_pix / area
    dic["frac_blue_area"] = blue_pix / area
    dic["radius"] = radius
    dic["area"] = np.pi * radius**2
    dic["x"] = x
    dic["y"] = y
    dic["frac_r_y"] = radius / y
    return dic
