import os
from os.path import join as pj
import sys
import datetime

import numpy as np
import pandas as pd
import pynmea2

from sign_detection import base_dir
from sign_detection.data import logger

def processGPS(s):
    if not (type(s) is str):
        return None
    if(not s.startswith("$GNGGA")):
        return None
    msg = pynmea2.parse(s)
    if msg.gps_qual==0:
        return None
    dic = {}
    dic["timestamp"] = msg.timestamp
    dic["latitude"] = msg.latitude
    dic["longitude"] = msg.longitude
    dic["horizontal_dil"] = msg.horizontal_dil
    return dic

def text2df(text_path, date):
    l = []
    with open(text_path, 'rU') as f:
        for line in f: 
            dic = processGPS(line)
            if dic is not None:
                #need to add date to time object
                time = dic["timestamp"]
                dt = datetime.datetime.combine(date,time)
                dic["timestamp"] = dt
                l.append(dic)
    df = pd.DataFrame(l)
    return df

def text2csv(text_path, csv_path, date):
    df = text2df(text_path, date)
    df.to_csv(csv_path, index=False)
    logger.info("Wrote {}".format(csv_path))

def text2pkl(text_path, pkl_path, date):
    df = text2df(text_path, date)
    df.to_pickle(pkl_path)
    logger.info("Wrote {}".format(pkl_path))

def main():
    in_path = args.in_path
    _, tail = os.path.split(in_path)
    name, _ = os.path.splitext(tail)
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    out_path_csv = pj(out_dir, name + ".csv")
    out_path_pkl = pj(out_dir, name + ".pkl")
    date = datetime.datetime.strptime(args.file_date, '%Y-%m-%d').date()
    text2csv(in_path, out_path_csv, date)
    text2pkl(in_path, out_path_pkl, date)

if __name__ == '__main__':

    from sign_detection import create_argument_parser
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[create_argument_parser()])

    # Inputs
    parser.add_argument('-in_path', 
                        default=pj(base_dir, "../data/raw/170329/gps/3.29.12.47_35.txt"),
                        type=str, help="Path to input text file.")
    parser.add_argument('-out_dir', 
                        default=pj(base_dir, "../data/interim/170329/gps"),
                        type=str, help="Path to output csv file.")
    parser.add_argument('-file_date', 
                        default="2017-03-29",
                        type=str, help="Date of file to add to datetime")
    args = parser.parse_args()

    logger.setLevel(args.logging_level)
    logger.debug("Arguments passed: {}".format(args))

    main()


