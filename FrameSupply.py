#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Framesupply classes used for reading out various formats and cameras.
"""

import threading
import cv2
import imageio
import numpy as np
from time import sleep
from PyQt5 import QtGui
import datetime
from scipy import ndimage

class FrameSupply:
    """
    Main class that can supply frames for further analysis.
    """

    def __init__(self):
        self.frameready = False
        self.is_running = False
        self.gotcapturetime=False
        self.framebuffer=[]
        self.nframes=0
        self.framenumber=int(0)

    def run(self):
        """
        Start the frame supply. Required to get frames.
        """
        pass

    def getnextframe(self):
        """
        Get the last frame from the frame supply buffer.
        Only possible if frameready is true.
        """
        pass
    def getframesize(self):
        """
        Get the width and height of the frame.
        """
        pass

class OpencvReadVideo(FrameSupply):
    """
    Read videofile with OpenCV
    """
    def __init__(self,VideoFile):
        super().__init__()
        self.VideoFile=VideoFile
        self.is_running = False
        self.gotcapturetime=False
        self.width = 0
        self.height = 0
    
    def start(self):
        self.cap = cv2.VideoCapture(self.VideoFile)
        self.is_running = True
        self.nframes=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def stop(self):
        """
        Stop the feed
        """
        self.cap.release()
        self.is_running = False
    
    def getfirstframe(self):
        ret, org_frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        return cv2.cvtColor(org_frame, cv2.COLOR_BGR2RGB),0
    
    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def getnextframe(self, step=1, number=None):
        # self.framenumber = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if hasattr(self, 'Set_framenumber'):
            self.framenumber = self.Set_framenumber
            del self.Set_framenumber
        elif number:
            self.framenumber = int(number)
        else:
            self.framenumber += step
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.framenumber)
        ret, org_frame = self.cap.read()
        if hasattr(self, 'FrameRotation'):
            org_frame = self.rotate_image(org_frame, self.FrameRotation)
        milliseconds_in_vid = self.cap.get(cv2.CAP_PROP_POS_MSEC )
        if ret:
            return cv2.cvtColor(org_frame, cv2.COLOR_BGR2RGB),self.framenumber, milliseconds_in_vid
        else:
            self.is_running=False
            self.stop()
            return -1,-1,-1
        
    def getframesize(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def start_writing(self):
        output_video = self.VideoFile[:-4] + '_compressed' + self.VideoFile[-4:]
        self.writer = cv2.VideoWriter(output_video, int(self.cap.get(cv2.CAP_PROP_FOURCC)), self.cap.get(cv2.CAP_PROP_FPS), (int(self.getframesize()[0]), int(self.getframesize()[1])))

    def stop_writing(self):
        self.writer.release()

    def append_frame(self, frame):
        self.writer.write(frame)
    

class ImageReader(FrameSupply):
    """
    Read videofile with OpenCV
    """
    def __init__(self,ImageFile):
        super().__init__()
        self.ImageFile=ImageFile
        self.is_running = False
        self.gotcapturetime=False
        self.framenumber=int(0)
        
    def start(self):
        self.is_running = True
        self.IOReader=imageio.get_reader(self.ImageFile)
        self.nframes=self.IOReader.get_length()
        
    def stop(self):
        """
        Stop the feed
        """
        self.is_running = False
        self.IOReader.close()
    
    def getfirstframe(self):
        org_frame=self.IOReader.get_data(0)
        return org_frame, 0
        
    def getnextframe(self):
        if self.framenumber<self.nframes:
            org_frame = self.IOReader.get_data(self.framenumber)
            self.framenumber+=1
            return org_frame,self.framenumber
        else:
            return -1,-1
        
    def getframesize(self):
        org_frame=self.IOReader.get_data(0)
        size=org_frame.shape
        return size[1],size[0]


class OpencvCamera(FrameSupply):
    """
    Camera operated using OpenCV
    """

    def __init__(self):
        super().__init__()
        self.framecaptime = []
        self.imaging_thread = []
        self.keep_running = False
        self.is_running = False
        self.gotcapturetime=True

    def start(self):
        """
        Start the camera
        """
        self.keep_running = True
        self.imaging_thread = threading.Thread(target=self._aquire)
        self.imaging_thread.start()

    def stop(self):
        """
        Stop the camera
        """
        self.keep_running = False
        

    def getnextframe(self):
        """
        Get the last frame
        :return:
        """
        self.nframes=len(self.framebuffer)
        if len(self.framebuffer)>=1:
            
            return self.framebuffer.pop(0),self.framecaptime.pop(0)
        else:
            return -1,-1
        
    def getframesize(self):
        if not 'self.cap' in locals():
            sleep(0.5)
            return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        else:
            return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def _aquire(self):
        if self.is_running:
            print("already running")
            return
        self.is_running = True
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened(): 
            errorpopup=QtGui.QMessageBox()
            errorpopup.setText('Error opening video stream')
            errorpopup.setStandardButtons(QtGui.QMessageBox.Ok)
            errorpopup.exec_()
            self.cap.release()
            self.is_running = False
            self.keep_running = False
        while self.keep_running:
            ret, org_frame = self.cap.read()
            self.framebuffer.append(cv2.cvtColor(org_frame, cv2.COLOR_BGR2RGB))
            self.framecaptime.append(np.datetime64(datetime.datetime.now()))
            self.frameready = True
        self.cap.release()
        self.is_running = False