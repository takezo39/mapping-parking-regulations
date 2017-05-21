import os
from os.path import join as pj
import sys
from datetime import datetime
import time
import shutil
from pprint import pformat

import cv2
import numpy as np
import pandas as pd
from skimage.draw import circle_perimeter
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
import imutils

import sign_detection
from sign_detection import base_dir
from sign_detection.features import logger
import sign_detection.features.detector as detector
from sign_detection.data.video_handler import VideoHandler

def main():

    logger.info("Your OpenCV version: {}".format(cv2.__version__))
    logger.info("Running on following cameras: {}".format(args.cameras2run))

    #circle finding parameters for garmin and canon cameras
    hough_radii_tups = [(25,150), (40,200)] #garmin/canon radii limits
    hough_thresholds = [0.2, 0.2] #garmin/canon relative accumulator thresholds

    # HSV thresholds for canon
    canon_red_thresholds = { 
        "hue_plus":30, #degrees relative to 0/180 degrees
        "hue_minus":40,
        "sat":70,
        "val":50
        }
    canon_blue_thresholds = { 
        "hue_plus":20, #degrees relative to 120 degrees
        "hue_minus":20,
        "sat":90,
        "val":50
        }

    # HSV thresholds for garmin
    garmin_red_thresholds = canon_red_thresholds 
    garmin_blue_thresholds = canon_blue_thresholds 

    # Set parameters passed as arguments
    hough_radii_increment = args.hough_radii_increment
    frame_increment = args.frame_increment
    frame_i = args.frame_i
    frame_f = args.frame_f
    flip_canon = args.flip_canon
    save_crops = args.save_crops
    del_old_images = args.del_old_images
    save_all_images = args.save_all_images
    display_images = args.display_images
    do_blue_cut = args.do_blue_cut
    dump_every_frame = args.dump_every_frame
    gps_path = args.gps_path

    #  Set input location of videos and video config files
    video_dir = pj(base_dir, "../data/raw", args.run_code, "video")

    #  Set location of output dir for images and dataframes
    if args.keep_output:
        out_dir = pj(base_dir, "../data/interim", args.run_code)
    else:
        out_dir = pj(base_dir, "../data/tmp", args.run_code)

    # Make image and dataframe dirs if they do not exist
    image_dir = pj(out_dir, "images")
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)
    df_dir = pj(out_dir, "dataframes")
    if not os.path.isdir(df_dir):
        os.makedirs(df_dir)

    # Clear the image directories for the cameras being processed
    red_dir_name = "red"
    blue_dir_name = "blue"
    product_dir_name = "product"
    crop_dir_name = "crop"
    img_dir_name = "img"
    img_dir_names = [
        red_dir_name,
        blue_dir_name,
        product_dir_name,
        crop_dir_name,
        img_dir_name]
    if (args.cameras2run == "BOTH" or args.cameras2run == "GARMIN")\
            and del_old_images:
        for name in img_dir_names:
            dir = pj(image_dir, "garmin", name)
            s = "Attempting to delete {} and to replace with empty dir"
            logger.debug(s.format(dir))
            if os.path.isdir(dir):
                shutil.rmtree(dir)
            os.makedirs(dir)
    if (args.cameras2run == "BOTH" or args.cameras2run == "CANON")\
            and del_old_images:
        for name in img_dir_names:
            dir = pj(image_dir, "canon", name)
            s = "Attempting to delete {} and to replace with empty dir"
            logger.debug(s.format(dir))
            if os.path.isdir(dir):
                shutil.rmtree(dir)
            os.makedirs(dir)

    # If ground_truth_file was passed as an argument, only process frames 
    # that have signs in them
    if args.ground_truth_file is not None:
        logger.info("Running around ground truth signs only")
        df_lab = pd.read_csv(args.ground_truth_file)
        frames = []
        num_frames = 24

        if args.cameras2run == "BOTH" or args.cameras2run == "GARMIN":
            garmin_frames = df_lab["garmin_base_frame"]
            for frame in garmin_frames:
                if frame != -1:
                    frames.extend(range(frame-num_frames, frame+num_frames+1, 
                                  frame_increment))

        if args.cameras2run == "BOTH" or args.cameras2run == "CANON":
            canon_frames = df_lab["canon_base_frame"]
            for frame in canon_frames:
                if frame != -1:
                    frames.extend(range(frame-num_frames, frame+num_frames+1, 
                                  frame_increment))
    else:
        s = "Running on frames {}:{}:{}"
        logger.info(s.format(frame_i, frame_increment, frame_f))
        frames = range(frame_i, frame_f+1, frame_increment)
    logger.debug("Processing following {} frames:".format(len(frames)))
    logger.debug(pformat(frames))

    #set up video handlers for garmin videos
    garmin_handlers = []
    if args.cameras2run == "BOTH" or args.cameras2run == "GARMIN":
        garmin_handlers = VideoHandler.handlers_from_config(video_dir, 
                                                            "garmin_config.csv",
                                                            gps_path)

    #set up video handlers for canon videos
    canon_handlers = []
    if args.cameras2run == "BOTH" or args.cameras2run == "CANON":
        canon_handlers = VideoHandler.handlers_from_config(video_dir, 
                                                           "canon_config.csv",
                                                           gps_path)

    # things to zip for for loop
    handlers = [garmin_handlers, canon_handlers]
    cameras = ["garmin", "canon"]
    red_thresholds = [garmin_red_thresholds, canon_red_thresholds]
    blue_thresholds = [garmin_blue_thresholds, canon_blue_thresholds]

    t0 = time.time()
    for handlers, camera, hough_radii_tup, hough_thresh, red_thresh, blue_thresh in \
            zip(handlers, cameras, hough_radii_tups, hough_thresholds, 
                red_thresholds, blue_thresholds):

        print "\n********* STARTING ALGORITHM FOR %s *********" % camera.upper()

        # define directory names to save images
        im_dir = pj(image_dir, camera)
        red_dir  = pj(im_dir, red_dir_name)
        blue_dir  = pj(im_dir, blue_dir_name)
        product_dir  = pj(im_dir, product_dir_name)
        crop_dir  = pj(im_dir, crop_dir_name)
        img_dir  = pj(im_dir, img_dir_name)

        # Set hough radii range for this camera
        hough_radii = np.arange(hough_radii_tup[0], hough_radii_tup[1], hough_radii_increment)

        all_features = [] # to hold all features of circles
        df_file_name = camera + "_" + str(frames[0]) + "_" + str(frames[-1]) + ".csv"
        df_file_path = pj(df_dir, df_file_name) # to save features
        total_circles = 0
        num_processed_frames = 0

        for base_frame in frames:
            print "\nbase_frame %07d" % base_frame
            ta = time.time()

            vh = VideoHandler.get_handler(handlers, base_frame)
            if vh is None:
                continue

            img = vh.get_frame(base_frame)
            if camera == "canon" and flip_canon:
                img = imutils.rotate(img, 180)
            coords = vh.get_frame_coords(base_frame=base_frame)

            #equalise histograms
            img = detector.equalise_value_local(img)
            img = detector.equalise_saturation_local(img)

            #blue-red edge detection
            red = detector.threshold_red(img, red_thresh)
            blue = detector.threshold_blue(img, blue_thresh)
            kernel = np.ones((10,10),np.uint8)
            blue_dilated = cv2.dilate(blue, kernel)
            product = np.multiply(blue_dilated, red) * 255 # go from 1 -> 255 for edge algorithm
            hough_res = hough_circle(product, hough_radii)

            circles = []
            accums = []

            for radius, h in zip(hough_radii, hough_res):
                peaks = peak_local_max(h, threshold_abs=hough_thresh)
                for peak in peaks:
                    circles.append([peak[1], peak[0], radius])
                    accums.append(h[peak[0], peak[1]])

            #remove overlapping circles (keeping strongest circle)
            circles, accums = detector.remove_overlap(circles, accums)

            # Make pixelated circles                                                   
            circles_ = list(np.uint16(np.around(circles)))

            # Remove circles that don't pass basic blue pixels cut
            if do_blue_cut:
                selection = [] 
                selection_ = []
                for c, c_ in zip(circles, circles_):
                    if detector.get_frac_blue_area(c, c_, blue) > 0.05:
                        selection.append(c)
                        selection_.append(c_)
                circles = selection
                circles_ = selection_

            # count  circles                                                           
            n_circles = len(circles)
            print "Found %2.0f circles" % (n_circles)              
            total_circles += n_circles                                                 

            # Loop through circles                                                     
            for i, (c, c_) in enumerate(zip(circles, circles_)):                   
                                                                                   
                # get features of circles                                          
                dic = detector.get_features(c, c_, red, blue)                               
                dic['frame_num'] = vh.base_frame2frame_num(base_frame)
                dic['base_frame'] = base_frame
                dic['id'] = "f%07d_x%04d_y%04d" % (base_frame, dic['x'], dic['y'])
                dic['lat'] = coords[0]                                              
                dic['lon'] = coords[1]                                              
                dic['accum'] = accums[i]
                all_features.append(dic)                                           
                                                                                   
                # save cropped images around circles                               
                if save_crops:                                                     
                    crop = detector.crop_ns_sign(img, c_)                                   
                    cropname = dic['id'] + '.jpg'
                    cv2.imwrite(os.path.join(crop_dir, cropname), crop)            

            #draw circles before displaying/saving whole image
            if display_images or save_all_images:                                          
                img = detector.draw_circles(circles_, img)                                            
                                                                                   
            if display_images:                                                         
                detector.display_image(img, "detected", True)                                   
                                                                                       
            filename = "base_frame%07d.jpg" %  base_frame
            if save_all_images:                                                            
                cv2.imwrite(os.path.join(red_dir, filename), red)                      
                cv2.imwrite(os.path.join(blue_dir, filename), blue)                    
                cv2.imwrite(os.path.join(product_dir, filename), 256*product)                      
                cv2.imwrite(os.path.join(img_dir, filename), img)                    

            #dump data 
            num_processed_frames += 1
            if dump_every_frame:
                freq = 1
            else:
                freq = 10
            if num_processed_frames % freq == 0:
                print "Saving csv"
                df = pd.DataFrame(all_features)                                                
                df.to_csv(df_file_path)

            tb = time.time()
            t = tb-ta
            print("time for loop: %s" % t)
                                                                                       
        print("total circles found for %s: %s" % (camera, total_circles))
        df = pd.DataFrame(all_features)                                                
        df.to_csv(df_file_path)

    t1 = time.time()
    t = t1 - t0
    print("total time taken was %s" % t)


if __name__ == '__main__':

    from sign_detection import create_argument_parser
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[create_argument_parser()])

    # Inputs
    parser.add_argument('-gps_path', 
                        default=pj(base_dir, "../data/interim/170329/gps/3.29.12.47_35.pkl"), 
                        type=str, help="Path to location of GPS file (gpx, pkl)")
    parser.add_argument('-frame_i', default=19200, type=int,
                        help="First frame to process")
    parser.add_argument('-frame_f', default=19699, type=int,
                        help="Last frame to process")
    parser.add_argument('-hough_radii_increment', default=1, type=int,
                        help="Increment for radius in hough circle search")
    parser.add_argument('-frame_increment', default=1, type=int,
                        help="Process only Nth frame")
    parser.add_argument('-run_code', default='170329', type=str, 
                        help="The run code defining where in/out files are/go")
    parser.add_argument('-ground_truth_file', default=None, type=str, 
                        help="Directory with canon and garmin videos")
    parser.add_argument('-cameras2run', default='BOTH',
                        choices=['BOTH', 'CANON', 'GARMIN'],
                        help='Which camera videos to process')
    parser.add_argument('-dont_flip_canon', dest='flip_canon', 
                        action='store_false',
                        help="Don't flip canon camera images before processing")
    parser.set_defaults(flip_canon=True)
    parser.add_argument('-dont_save_crops', dest='save_crops', 
                        action='store_false',
                        help="Don't save cropped images of sign candidates")
    parser.set_defaults(save_crops=True)
    parser.add_argument('-dont_del_old', dest='del_old_images', 
                        action='store_false', help="Regenerate image folders")
    parser.set_defaults(del_old_images=True)
    parser.add_argument('-save_all_images', dest='save_all_images', 
                        action='store_true', help="Save all generated images")
    parser.set_defaults(save_all_images=False)
    parser.add_argument('-display_images', dest='display_images', 
                        action='store_true', help="Display images each loop")
    parser.set_defaults(display_images=False)
    parser.add_argument('-no_blue_cut', dest='do_blue_cut', 
                        action='store_false',
                        help="Don't require min 5% blue pixels in sign candidates")
    parser.set_defaults(do_blue_cut=True)

    # Outputs
    parser.add_argument('-dump_every_frame', dest='dump_every_frame', 
                        action='store_true',
                        help="Dump feature dataframe every frame")
    parser.set_defaults(dump_every_frame=False)
    parser.add_argument('-keep_output', dest='keep_output', 
                        action='store_true',
                        help="Keep output images and dataframes in project dir")
    parser.set_defaults(dump_every_frame=False)

    args = parser.parse_args()

    logger.setLevel(args.logging_level)
    sign_detection.features.logger.setLevel(args.logging_level)
    sign_detection.data.logger.setLevel(args.logging_level)
    logger.debug("Arguments passed: {}".format(args))

    main()


