from binascii import Error
import numpy as np
from numpy.core.numeric import NaN
import scipy as sp


def linear_subpixel_detection(image, thresh, drop_center=0, old_left_arr=[], old_right_arr=[], mode=0):
    if mode == 0:
        return linear_subpixel_detection_from_center(image, thresh, drop_center, old_left_arr, old_right_arr)
    else:
        return linear_subpixel_detection_from_edge(image, thresh, old_left_arr, old_right_arr)


def linear_subpixel_detection_from_center(image, thresh, drop_center, old_left_arr=[], old_right_arr=[]):
    framesize=image.shape
    edgeleft=np.zeros(framesize[0])
    edgeright=np.zeros(framesize[0])
    edgeleft_old = NaN
    edgeright_old = NaN

    if len(old_left_arr) != 0:
        search_radius = 15
    else:
        search_radius = 10

    for y in range(framesize[0]): # edge detection, go line by line on horizontal from top to bottom of image
        try: # get left edge
            if len(old_left_arr) != 0: # search in neighborhood of edge from previous frame
                edgeleft_old = old_left_arr[y]
            detect_arr = (image[y,int(edgeleft_old-search_radius):int(edgeleft_old+search_radius)]>thresh)[::-1]
            if np.mean(detect_arr) == 0: # nothing detected
                raise Error
            edgeleft[y] = int(edgeleft_old + search_radius) - np.argmax(detect_arr)
        except: # search in whole image for first white pixel beginning from center of drop to left edge of image
            edgeleft[y] = drop_center - np.argmax((image[y,0:drop_center]>thresh)[::-1])
        edgeleft_old = edgeleft[y]
        try: # subpixel correction with using corr=(threshold-intensity(edge-1))/(intensity(edge)-intensity(edge-1))
            if np.float_(image[y,np.int(edgeleft[y])]) != np.float_(image[y,np.int(edgeleft[y]-1)]):
                subpxcorr = (thresh-np.float_(image[y,np.int(edgeleft[y]-1)]))/(np.float_(image[y,np.int(edgeleft[y])])-np.float_(image[y,np.int(edgeleft[y]-1)]))
                if abs(subpxcorr) < 1.1:
                    edgeleft[y] = edgeleft[y] + subpxcorr - 1 #add the correction and shift 1 pixel left, to plot on the edge properly
        except:
            pass

        try: # get right edge
            if len(old_right_arr) != 0:
                edgeright_old = old_right_arr[y]
            detect_arr = (image[y,int(edgeright_old-search_radius):int(edgeright_old+search_radius)]>thresh)
            if np.mean(detect_arr) == 0: # nothing detected
                raise Error
            edgeright[y] = int(edgeright_old - search_radius) + np.argmax(detect_arr)
        except: # search in whole image for first white pixel beginning from center of drop to right edge of image
            edgeright[y] = drop_center + np.argmax((image[y,drop_center:framesize[1]]>thresh))
        edgeright_old = edgeright[y]
        # same scheme for right edge, except the edge detection is done flipped, since np.argmax gives the first instance of the maximum value
        try:
            if np.float_(image[y,np.int(edgeright[y])]) != np.float_(image[y,np.int(edgeright[y]-1)]):
                subpxcorr = (thresh-np.float_(image[y,np.int(edgeright[y]-1)]))/(np.float_(image[y,np.int(edgeright[y])])-np.float_(image[y,np.int(edgeright[y]-1)]))
                if abs(subpxcorr) < 1.1:
                    edgeright[y] = edgeright[y]+subpxcorr-1
        except:
            pass

    edgeleft[edgeleft == np.inf] = framesize[1]
    edgeright[edgeright == np.inf] = framesize[1]
    edgeleft[edgeleft == -np.inf] = 0
    edgeright[edgeright == -np.inf] = 0
    return edgeleft, edgeright;


def linear_subpixel_detection_from_edge(image, thresh, old_left_arr=[], old_right_arr=[]):
    framesize=image.shape
    edgeleft=np.zeros(framesize[0])
    edgeright=np.zeros(framesize[0])
    edgeleft_old = NaN
    edgeright_old = NaN

    if len(old_left_arr) != 0:
        search_radius = 15
    else:
        search_radius = 10

    for y in range(framesize[0]): # edge detection, go line by line on horizontal from top to bottom of image
        try: # get left edge
            if len(old_left_arr) != 0: # search in neighborhood of edge from previous frame
                edgeleft_old = old_left_arr[y]
            detect_arr = (image[y,int(edgeleft_old-search_radius):int(edgeleft_old+search_radius)] < thresh)
            if (np.mean(detect_arr) == 0) or (np.mean(detect_arr) == 1): # nothing detected
                raise Error
            edgeleft[y] = int(max(edgeleft_old-search_radius, 0)) + np.argmax(detect_arr)
        except: # search in whole image for first dark pixel beginning from left edge of image
            edgeleft[y] = np.argmax(image[y]<thresh)
        edgeleft_old = edgeleft[y]
        try: # subpixel correction with using corr=(threshold-intensity(edge-1))/(intensity(edge)-intensity(edge-1))
            if np.float_(image[y,np.int(edgeleft[y])]) != np.float_(image[y,np.int(edgeleft[y]-1)]):
                subpxcorr = (thresh-np.float_(image[y,np.int(edgeleft[y]-1)]))/(np.float_(image[y,np.int(edgeleft[y])])-np.float_(image[y,np.int(edgeleft[y]-1)]))
                if abs(subpxcorr) < 1.1:
                    edgeleft[y] = edgeleft[y]+subpxcorr-1 #add the correction and shift 1 pixel left, to plot on the edge properly
        except:
            pass

        try: # get right edge
            if len(old_right_arr) != 0:
                edgeright_old = old_right_arr[y]
            detect_arr = (image[y,int(edgeright_old-search_radius):int(edgeright_old+search_radius)] < thresh)[::-1]
            if (np.mean(detect_arr) == 0) or (np.mean(detect_arr) == 1): # nothing detected
                raise Error
            edgeright[y] = int(min(edgeright_old+search_radius, framesize[1])) - np.argmax(detect_arr)
        except: # search in whole image for first dark pixel beginning from right edge of image
            edgeright[y] = (framesize[1] - np.argmax((image[y]<thresh)[::-1]))
        edgeright_old = edgeright[y]
        # same scheme for right edge, except the edge detection is done flipped, since np.argmax gives the first instance of the maximum value
        try:
            if np.float_(image[y,np.int(edgeright[y])]) != np.float_(image[y,np.int(edgeright[y]-1)]):
                subpxcorr = (thresh-np.float_(image[y,np.int(edgeright[y]-1)]))/(np.float_(image[y,np.int(edgeright[y])])-np.float_(image[y,np.int(edgeright[y]-1)]))
                if abs(subpxcorr) < 1.1:
                    edgeright[y] = edgeright[y]+subpxcorr-1
        except:
            pass

    edgeleft[edgeleft == np.inf] = framesize[1]
    edgeright[edgeright == np.inf] = framesize[1]
    edgeleft[edgeleft == -np.inf] = 0
    edgeright[edgeright == -np.inf] = 0
    return edgeleft, edgeright;


def errorfunction_subpixel_detection(image,thresh):
    erffitsize=np.int(40)
    def errorfunction(x,xdata,y): #define errorfunction to fit with a least squares fit.
        return x[0]*(1+sp.special.erf(xdata*x[1]+x[2]))+x[3] - y;
    framesize=image.shape
    edgeleft=np.zeros(framesize[0])
    edgeright=np.zeros(framesize[0])
    for y in range(framesize[0]-1): #edge detection, go line by line on horizontal
        edgeleft[y]=np.argmax(image[y,0:framesize[1]]<thresh) #edge detection on pixel scale
        if (edgeleft[y]-erffitsize)>=0  and (edgeleft[y]-erffitsize)<=framesize[0]:
            fitparts=np.array(image[y,range(np.int(edgeleft[y])-erffitsize,np.int(edgeleft[y])+erffitsize)]) #take out part of the image around the edge to fit the error function
            guess=(max(fitparts)-min(fitparts))/2,-.22,0,min(fitparts) #initial guess for error function
            lstsqrsol=sp.optimize.least_squares(errorfunction,guess,args=(np.array(range(-erffitsize,erffitsize)),fitparts)) #least sqaures fit
            edgeleft[y]=edgeleft[y]-lstsqrsol.x[2]/lstsqrsol.x[1] #add the subpixel correction
        edgeright[y]=np.int(framesize[1]-np.argmax(image[y,range(framesize[1]-1,0,-1)]<thresh))#same scheme for right edge, except the edge detection is done flipped, since np.argmax gives the first instance of the maximum value
        if (edgeright[y]-erffitsize)>=0  and (edgeright[y]-erffitsize)<=framesize[0]:
            fitparts=np.array(image[y,range(np.int(edgeright[y])-erffitsize,np.int(edgeright[y])+erffitsize)])
            guess=(max(fitparts)-min(fitparts))/2,.22,0,min(fitparts)
            lstsqrsol=sp.optimize.least_squares(errorfunction,guess,args=(np.array(range(-erffitsize,erffitsize)),fitparts))
            edgeright[y]=edgeright[y]-lstsqrsol.x[2]/lstsqrsol.x[1]
        elif edgeright[y]==framesize[1]:
            edgeright[y]=0

    return edgeleft, edgeright;
