from logging import FileHandler
import FrameSupply
import magic
import sys, os, glob
import cv2
import numpy as np
import pandas as pd


filetypemap={'image/tiff':FrameSupply.ImageReader,'image/jpeg':FrameSupply.ImageReader,'image/png':FrameSupply.ImageReader,'video/x-msvideo':FrameSupply.OpencvReadVideo}


def get_video_file(path):
    """Open the Video File the corresponds to 'path' and get the first frame"""
    VideoFile = path
    mimetype = magic.from_file(VideoFile,mime=True)
    if VideoFile[-3:] == 'MOV':
        mimetype = 'video/x-msvideo'
    if VideoFile[-3:] == 'avi':
        mimetype = 'video/x-msvideo'
    if VideoFile[-3:] == 'MP4':
        mimetype = 'video/x-msvideo'

    if any(mimetype in key for key in filetypemap):
        FrameSource = filetypemap[mimetype](VideoFile)
        FrameSource.start()
        firstframe,_ = FrameSource.getfirstframe()

    return FrameSource


def compress_video(video_path, comp_level, path_top, path_result, chosen_drop):
    print("Start writing new video file of compression level {} for {}.".format(comp_level, chosen_drop))
    try:
        datafile = os.path.join(path_result, chosen_drop+'.xlsx')
        data = pd.read_excel(datafile, header=0, index_col=0)
    except:
        try: # read old text file
            datafile = os.path.join(os.path.join(path_result, chosen_drop), chosen_drop+'.txt')
            data = pd.read_csv(datafile, header=0, sep='    ')
        except:
            raise ValueError('No result file found!')

    if video_path[-15:-5] == "compressed": # compress already compressed video further
        old_comp_level = int(video_path[-5])
        if old_comp_level >= comp_level:
            raise ValueError("The video is already compressed to at least your desired compressioin level")
        
    FrameSource = get_video_file(video_path)
    output_name = chosen_drop + '_compressed' + str(comp_level) + FrameSource.VideoFile[-4:]
    output_video = os.path.join(os.path.join(path_top, chosen_drop), output_name)

    writer = cv2.VideoWriter(output_video, int(FrameSource.cap.get(cv2.CAP_PROP_FOURCC)), FrameSource.cap.get(cv2.CAP_PROP_FPS), (int(FrameSource.getframesize()[0]), int(FrameSource.getframesize()[1])))
    # indentify lines that correspond to frames used for new compressed video
    indexes = (data['weight']>=comp_level)
    framenumber = data['framenumber'].to_numpy()[indexes]
    captime = 0

    if video_path[-15:-5] == "compressed":
        old_weight = data['weight'][data['weight']>=old_comp_level]
        # convert framenumbers of original video to framenumbers in already compressed video
        framenumber = np.arange(1, np.size(old_weight)+1, dtype=int)
        # indentify lines that correspond to frames used for new compressed video
        indexes = (old_weight>=comp_level)
        framenumber = framenumber[indexes]

    for i in range(np.size(framenumber)):
        if (framenumber[i]-captime) > FrameSource.nframes: # end of original video, next video of the same drop is used
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


def create_video(path_top, path_result, chosen_drop, comp_level=0):
    """Stack analysed frames of all videos of one drop together to compressed video of chosen compression level 'comp_level'
    (by default all compression levels will be produced)"""
    os.chdir(path_top)
    video_endings = ["MOV", "MP4"]

    if comp_level not in range(4):
        raise ValueError("No possible compression level.")

    if comp_level == 0:
        print("Start writing new video files of all compression levels.")
        for i in range(1, 4):
            create_video(path_top, path_result, chosen_drop, comp_level=i)

    elif chosen_drop == 'All':
        success = True
        drop_no = 1
        while success:
            success = create_video(path_top, path_result, chosen_drop='Drop_'+str(drop_no), comp_level=comp_level)
            drop_no += 1
        return True

    else:
        for drop in glob.glob("*/"):
            if drop[:-1] == chosen_drop:
                os.chdir(drop)
                for ending in video_endings:
                    for video in glob.glob("*."+ending):
                        compress_video(video, comp_level, path_top, path_result, chosen_drop)
                        os.chdir('..')
                        return True
        return False

if __name__ == '__main__':
    top_path = sys.argv[1]
    drop = sys.argv[2]
    level = int(sys.argv[3])
    result_path = sys.argv[4]
    success = create_video(top_path, result_path, drop, level)
