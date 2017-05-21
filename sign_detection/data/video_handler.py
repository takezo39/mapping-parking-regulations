from os.path import join as pj

import numpy as np
import pandas as pd
import cv2
import gpxpy
import datetime
from geopy.distance import vincenty
import imutils

from sign_detection import base_dir
from sign_detection.data import logger

class VideoHandler():

    def __init__(self, videopath, fps, gps_path, sync_tup, base_frame_limits,
                 gps_time_res=0.1):

        self.capturer = self.get_capturer(videopath)
        if gps_path.lower().endswith(".gpx"):
            self.gpsdic = self._gpx2gpsdic(gps_path)
        elif gps_path.lower().endswith(".pkl"):
            self.gpsdic = self._pkl2gpsdic(gps_path)
        else:
            raise TypeError("{} must be .gpx or .pkl file".format(gps_path))
        self.gps_time_res = gps_time_res
        self.frame_period = 1./fps
        self.start_time = self.get_gmt_start(sync_tup[0], sync_tup[1])
        self.base_frame_limits = base_frame_limits

    @staticmethod
    def handlers_from_config(video_dir, config_file, gps_path):
        """Return multiple VideoHandlers for each row in config file

        Args:
            video_dir(str): The dir containing the configuration file and videos
            config_file(str): The configuration file

        Returns:
           handlers: list of VideoHandler
        """

        #  First get sync variables for videos from sync_tup.csv in video_dir
        sync_tup_path = pj(video_dir, "sync_tup.csv")
        try:
            df_sync_tup = pd.read_csv(sync_tup_path)
        except IOError:
            logger.error("No file found at {}".format(sync_tup_path))
            raise
        sync_time_str = df_sync_tup["sync_time"].iloc[0]
        sync_time = datetime.datetime.strptime(sync_time_str, 
                                               '%Y-%m-%d %H:%M:%S.%f')
        sync_frame_base = df_sync_tup["sync_frame_base"].iloc[0]

        #  Now create list of handlers from data in config_file
        handlers = []
        videopaths, frame_limits, fpss = VideoHandler._parse_config(video_dir, 
                                                                    config_file)
        for path, frame_limits, fps in zip(videopaths, frame_limits, fpss):
            sync_frame = sync_frame_base - frame_limits[0]
            sync_tup = (sync_frame, sync_time)
            handlers.append(VideoHandler(path, fps, gps_path, sync_tup, 
                                         frame_limits))

        return handlers

    @staticmethod
    def _parse_config(video_dir, config_file):
        """Parse a configuration csv file for multiple videos

        Args:
            video_dir(str): The dir containing the configuration file and videos
            config_file(str): The configuration file

        Returns:
           videopaths: list with paths to videos
           frame_limits: list of tuples of the start and end frames wrt baseline
        """
 
        df = pd.read_csv(pj(video_dir, config_file))
        videopaths = [ pj(video_dir, f) for f in list(df["videos"]) ]
        start_frames = np.array(df["start_frames"])
        lengths = np.array(df["lengths"])
        fpss = np.array(df["fps"])
        end_frames = start_frames + lengths - 1
        frame_limits = []
        for i, f in zip(start_frames, end_frames):
            frame_limits.append((i,f))
        return videopaths, frame_limits, fpss

    @staticmethod
    def get_handler(handlers, base_frame):
        """Get the correct handler for a given base_frame from a list of handlers 

        Args:
            handlers(list of VideoHandler)
            base_frame(int): frame of interest

        Returns:
            handler(VideoHandler)
        """
        
        vh = None
        for handler in handlers:
            frame_limits = handler.base_frame_limits
            if base_frame >= frame_limits[0] and base_frame <= frame_limits[1]:
                vh = handler
                handler_found = True
                break
        if vh is None:
            logger.info("No handler found for base_frame={}".format(base_frame))

        return vh

    def get_capturer(self, videopath):
        cap = cv2.VideoCapture(videopath)
        while not cap.isOpened():
            cap = cv2.VideoCapture(videopath)
            cv2.waitKey(1000)
            print "Wait for the header"
        return cap

    def get_frame(self, base_frame):
        """Returns the frame corresponding to base_frame from Blender project

        Args:
            base_frame(int)

        Returns:
            numpy array of frame

        """
        logger.debug("Getting image for base_frame = {}".format(base_frame))

        frame_num = self.base_frame2frame_num(base_frame)
        if imutils.is_cv2():
            self.capturer.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_num)
        elif imutils.is_cv3():
            self.capturer.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        flag, img = self.capturer.read()                                                   
        
        while flag is False:
            print "frame is not ready"                                             
            cv2.waitKey(1000)          
            flag, img = self.capturer.read()                                                   

        return img

    def base_frame2frame_num(self, base_frame):
        frame_num =  base_frame - self.base_frame_limits[0]
        s = "base_frame:{} corresponds to frame_num:{}"
        logger.debug(s.format(base_frame, frame_num))
        return frame_num

    def _gpx2gpsdic(self, gpx_path):
        """Returns a {time:{lat:, lon:}} dictionary for a gpx file

        Args:
            gpx_path: Path to gpx file

        Returns:
            dictionary as described above

        """

        gpx = self._parse_gpx(gpx_path)
        gpsdic = {}
	for track in gpx.tracks:
	    for segment in track.segments:
		for point in segment.points:
                    pt = {"lat": point.latitude, "lon": point.longitude}
                    gpsdic[point.time] = pt
        return gpsdic

    def _pkl2gpsdic(self, pkl_path):
        """Returns a {time:{lat:, lon:}} dictionary for a pkl file

        Args:
            pkl_path: Path to pkl file

        Returns:
            dictionary as described above

        """

        df = pd.read_pickle(pkl_path)
        gpsdic = {}
        for _, row in df.iterrows():
            latlondic = {"lat":row["latitude"], "lon":row["longitude"]}
            gpsdic[row["timestamp"]] = latlondic
        return gpsdic


    def _parse_gpx(self, gpxpath):
        """Parses a gps file

        Args:
            gpxpath (str): full path to gpx file

        Returns:
            A gpxpy GPX object (parse of a gpx file)

        """

        with open(gpxpath, 'rb') as f:
            gpx = gpxpy.parse(f)
        return gpx

    def get_gpx_timeres(self, gpx):
        """Gets time resolution of GPX file in seconds
        TODO: delete this function once you stop using GPX

        Note:
            Assumes gpx.tracks[0].segements[0] exists and that ~.points[0]
            and ~.points[1] are consequtive. Also that resolution is constant

        Args:
            gpx: a gpxpy GPX object (parse of a gpx file)

        Returns:
            resolution in seconds

        """

        t0 = gpx.tracks[0].segments[0].points[0].time
        t1 = gpx.tracks[0].segments[0].points[1].time
        return (t1 - t0).seconds

    def get_gmt_start(self, frame_num, frame_gmt):
        """Sets the start time of video in GMT 

        Note:
            Assumes video starts at frame zero

        Args:
            frame_num (int): a frame number
            frame_gmt (datetime.datetime):  GMT time for frame_num

        Returns:
            datetime object

        """
        delta = datetime.timedelta(seconds = frame_num * self.frame_period)
        return frame_gmt - delta

    def date2secs(self, time):
        """Convert datetime to seconds

        Args:
            time (datetime)

        Returns:
            number in seconds

        """
        return (time-datetime.datetime(1970,1,1)).total_seconds()

    def get_frame_gmt(self, frame_num):
        """Gets the time of a frame in GMT 

        Note:
            Assumes video starts at frame zero

        Args:
            frame_num (int): a frame number

        Returns:
            datetime object

        """
        delta = datetime.timedelta(seconds = frame_num * self.frame_period)
        return self.start_time + delta

    def get_next_gps(self, time):
        """Gets GPS coordinates from self.gpsdic for next time step wrt time

        Args:
            time (datetime)

        Returns:
            latitude, longitude, datetime

        """

        dummydate = datetime.datetime(1970, 1, 1)
        time_in_secs = (time - dummydate).total_seconds()
        delta = self.gps_time_res - (time_in_secs % self.gps_time_res)
        next_time = time + datetime.timedelta(seconds=delta)
        found_gps = False
        skipped_time = 0
        while not found_gps:
            # keep stepping through time until you find a dictionary entry
            try:
                gps = self.gpsdic[next_time]
            except KeyError:
                next_time += datetime.timedelta(seconds=self.gps_time_res)
                skipped_time += self.gps_time_res
            else:
                found_gps = True
            if skipped_time > 1:
                raise KeyError("Could not find next gps time within 1 sec")

        return gps["lat"], gps["lon"], next_time

    def get_last_gps(self, time):
        """Gets GPS coordinates from self.gpsdic for last time step wrt time

        Args:
            time (datetime)

        Returns:
            latitude, longitude, datetime

        """

        dummydate = datetime.datetime(1970, 1, 1)
        time_in_secs = (time - dummydate).total_seconds()
        delta = time_in_secs % self.gps_time_res
        last_time = time - datetime.timedelta(seconds=delta)

        found_gps = False
        skipped_time = 0
        while not found_gps:
            # keep stepping through time until you find a dictionary entry
            try:
                gps = self.gpsdic[last_time]
            except KeyError:
                last_time -= datetime.timedelta(seconds=self.gps_time_res)
                skipped_time += self.gps_time_res
            else:
                found_gps = True
            if skipped_time > 1:
                raise KeyError("Could not find last gps time within 1 sec")

        return gps["lat"], gps["lon"], last_time

    def get_frame_coords(self, frame_num=None, base_frame=None):
        """Interpolate the coordinates for a given frame from 2 closest points

        Args:
            Should only pass one of:
            frame_num(int): frame no. of video
            base_frame(int): frame no. from Bledner project

        Returns:
            latitude, longitude

        """
        if frame_num is None and base_frame is None:
            print("ERROR: one of frame_num or base_frame must be and int")
        elif frame_num is not None and base_frame is not None:
            print("ERROR: one of frame_num or base_frame must be and int")
        elif base_frame is not None:
            frame_num = self.base_frame2frame_num(base_frame)

        time = self.get_frame_gmt(frame_num)
        secs = self.date2secs(time)

        lat_last, lon_last, time_last = self.get_last_gps(time)
        secs_last = self.date2secs(time_last)
        lat_next, lon_next, time_next = self.get_next_gps(time)
        secs_next = self.date2secs(time_next)

        lat = np.interp(secs, [secs_last, secs_next], [lat_last, lat_next])
        lon = np.interp(secs, [secs_last, secs_next], [lon_last, lon_next])

        return (lat, lon)

    def distance(self, p1, p2):
        """Calculate distance betwween two points

        Args:
            p1((float, float) tuple): latitude, longitude in decimals
            p2((float, float) tuple): latitude, longitude in decimals

        Returns:
            distance in metres

        """

        return vincenty(p1, p2).meters






