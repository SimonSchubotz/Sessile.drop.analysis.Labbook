from logging import FileHandler
import FrameSupply
import magic
import sys, os, glob
import cv2
import numpy as np


filetypemap={'image/tiff':FrameSupply.ImageReader,'image/jpeg':FrameSupply.ImageReader,'image/png':FrameSupply.ImageReader,'video/x-msvideo':FrameSupply.OpencvReadVideo}


def get_video_file(path):
    """Open the Video File the corresponds to 'path' and get the first frame"""
    VideoFile = path
    mimetype = magic.from_file(VideoFile,mime=True)
    if VideoFile[-3:] == 'MOV':
        mimetype = 'video/x-msvideo'
    if VideoFile[-3:] == 'avi':
        mimetype = 'video/x-msvideo'

    if any(mimetype in key for key in filetypemap):
        FrameSource = filetypemap[mimetype](VideoFile)
        FrameSource.start()
        firstframe,_ = FrameSource.getfirstframe()

    return FrameSource


def compress_video(video_path, comp_level):
    if len(glob.glob("*.txt"))<1:
        raise ValueError('No text file found!')
    
    for text in glob.glob("*.txt"):
        if (text[0:5] == 'Drop_'):
            textfile = text

    if video_path[-15:-5] == "compressed": # compress already compressed video further
        old_comp_level = int(video_path[-5])
        if old_comp_level >= comp_level:
            raise ValueError("The video is already compressed to at least your desired compressioin level")
        FrameSource = get_video_file(video_path)
        output_video = FrameSource.VideoFile[:-5] + str(comp_level) + FrameSource.VideoFile[-4:]

    elif video_path[:3] == "DSC": # compress original video
        FrameSource = get_video_file(video_path)
        output_video = FrameSource.VideoFile[:-12]+FrameSource.VideoFile[-19:-13] + '_compressed' + str(comp_level) + FrameSource.VideoFile[-4:]

    else:
        raise ValueError('No accepted video')

    writer = cv2.VideoWriter(output_video, int(FrameSource.cap.get(cv2.CAP_PROP_FOURCC)), FrameSource.cap.get(cv2.CAP_PROP_FPS), (int(FrameSource.getframesize()[0]), int(FrameSource.getframesize()[1])))
    framenumber, time, weight= np.loadtxt(textfile, skiprows=1, unpack=True, delimiter="    ", dtype={'names': ('framenumber', 'time', 'weight'), 'formats': ('i4', 'U26', 'i4')})
    # delete lines that correspond to frames not used for new compressed video
    indexes = (weight>=comp_level) + (framenumber==0)
    framenumber, time, weight = framenumber[indexes], time[indexes], weight[indexes]
    captime = 0

    if video_path[-15:-5] == "compressed":
        # convert framenumbers of original video to framenumbers in already compressed video
        framenumber = np.arange(1, np.size(framenumber)+1, dtype=int)
        # delete lines that correspond to frames not used for new compressed video
        framenumber, time, weight = framenumber[weight>=comp_level], time[weight>=comp_level], weight[weight>=comp_level]

    for i in range(np.size(framenumber)):
        if framenumber[i] == 0: # end of original video, next video of the same drop is used
            try:
                captime += FrameSource.nframes
                FrameSource.stop()
                video_path = video_path[:-8] + str(int(video_path[-8:-4])+1).zfill(4) + video_path[-4:] # number in video name increased by one (filled up with possible zeros)
                FrameSource = get_video_file(video_path)
                print("Using next video of drop.")
            except:
                break
        else:
            org_frame, framecaptime, milliseconds_in_vid = FrameSource.getnextframe(number=framenumber[i]-captime)
            writer.write(org_frame)

    writer.release()
    print("New video file complete.")
    os.chdir('..')


def create_video(path_top, chosen_drop, comp_level=0):
    """Stack analysed frames of all videos of one drop together to compressed video of chosen compression level 'comp_level'
    (by default all compression levels will be produced)"""
    os.chdir(path_top)

    if comp_level not in range(4):
        raise ValueError("No possible compression level.")

    if comp_level == 0:
        print("Start writing new video files of all compression levels.")
        for i in range(1, 4):
            create_video(path_top, chosen_drop, comp_level=i)

    else:
        print("Start writing new video file of compression level {}.".format(comp_level))

        for drop in glob.glob("*/"):
            if chosen_drop == 'All':
                if drop[:5] == 'Drop_':
                    os.chdir(drop)
                    for video in glob.glob("*.MOV"):
                        compress_video(video, comp_level)
                        return
            
            else:
                if drop[:-1] == chosen_drop:
                    os.chdir(drop)
                    for video in glob.glob("*.MOV"):
                        compress_video(video, comp_level)
                        return

if __name__ == '__main__':
    top_path = sys.argv[1]
    drop = sys.argv[2]
    level = int(sys.argv[3])
    create_video(top_path, drop, level)
