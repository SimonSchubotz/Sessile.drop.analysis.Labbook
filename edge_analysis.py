from shapely.geometry import LineString
import numpy as np


def analysis(edgeleft, edgeright, baseinput, framesize, k=100, PO_left=2, PO_right=2, reflection=True, BaselineFixed = False):
    """
    Analyzes the detected edge of the drop with the set baseline to
    give the contact angle, contact line position, and drop volume
    k represents the number of pixels up from the baseline to fit, PO is the 
    order of the polyfit used to fit the edge of the drop
    Returns contactpoints left and right, theta left and right, and drop volume
    """
    def get_intersect(edge, contact_ind, ind_low, ind_high):
        # linear fits for edge of drop
        upper_a, upper_b = np.polyfit(edge[ind_low:contact_ind], (range(ind_low,contact_ind)), 1)
        lower_a, lower_b = np.polyfit(edge[contact_ind:ind_high], (range(contact_ind,ind_high)), 1)
        # extension of fits
        x_1 = edge[ind_low]-500
        x_2 = edge[ind_high]+500
        upper_base = LineString([(x_1, get_y(upper_a, upper_b, x_1)), (x_2, get_y(upper_a, upper_b, x_2))]) 
        lower_base = LineString([(x_1, get_y(lower_a, lower_b, x_1)), (x_2, get_y(lower_a, lower_b, x_2))])
        # contact point of drop and surface (intersection of fitted upper and lower edge lines) 
        contact = upper_base.intersection(lower_base)
        return contact

    def get_y(a, b, x):
        return a*x+b

    baseline = LineString(baseinput) 
    rightline = LineString(np.column_stack((edgeright,(range(0,framesize[0]))))) 
    leftline = LineString(np.column_stack((edgeleft,(range(0,framesize[0])))))

    leftcontact = baseline.intersection(leftline) # approximation of contact points (intersection of linear splines)
    rightcontact = baseline.intersection(rightline)
    if leftcontact.geom_type == 'MultiPoint':
        leftcontact = leftcontact[-1]
    if rightcontact.geom_type == 'MultiPoint':
        rightcontact = rightcontact[0]

    base_corridor = 10 # corridor around baseline that is used for edge fitting
    basevec = np.array([-(baseinput[1,1]-baseinput[0,1])/baseinput[1,0],1])

    if not BaselineFixed:
        if reflection: # get contactpoints from intersection of linear fits of drop edge and reflection
            # linear fit of right edge close to baseline
            ind_low = max(int(rightcontact.y)-base_corridor, 0) # indexes of corridor for fit
            ind_high = min(int(rightcontact.y)+base_corridor, edgeright.size)
            points_clos_base = edgeright[ind_low:ind_high]
            if (np.argmax(points_clos_base) > 2) and (np.argmax(points_clos_base) < (len(points_clos_base)-3)): # contactpoint = point where drop edge is most right (for contact angle < 90°)
                contact_ind = int(np.argmax(points_clos_base) + int(rightcontact.y) - base_corridor)
            elif (np.argmin(points_clos_base) > 2) and (np.argmin(points_clos_base) < (len(points_clos_base)-3)): # contactpoint = point where drop edge is most left (for contact angle > 90°)
                contact_ind = int(np.argmin(points_clos_base) + int(rightcontact.y) - base_corridor)
            else: # contactpoint = center of search corridor (for contact angle ~ 90°)
                contact_ind =  int(rightcontact.y)
            rightcontact = get_intersect(edgeright, contact_ind, ind_low, ind_high)
            rightcontact_y = rightcontact.y

            # linear fit of left edge close to baseline
            ind_low = max(int(leftcontact.y)-base_corridor, 0)
            ind_high = min(int(leftcontact.y)+base_corridor, edgeleft.size)
            points_clos_base = edgeleft[ind_low:ind_high]
            if (np.argmin(points_clos_base) > 2) and (np.argmin(points_clos_base) < (len(points_clos_base)-3)): # contactpoint = point where drop edge is most left (for contact angle < 90°)
                contact_ind = int(np.argmin(points_clos_base) + int(leftcontact.y) - base_corridor) 
            elif (np.argmax(points_clos_base) > 2) and (np.argmax(points_clos_base) < (len(points_clos_base)-3)): # contactpoint = point where drop edge is most right (for contact angle > 90°)
                contact_ind = int(np.argmax(points_clos_base) + int(leftcontact.y) - base_corridor)
            else: # contactpoint = center of search corridor (for contact angle ~ 90°)
                contact_ind =  int(leftcontact.y)
            leftcontact = get_intersect(edgeleft, contact_ind, ind_low, ind_high)
            leftcontact_y = leftcontact.y
        
        else: # get contactpoints from jump of edge at baseline
            diff = np.diff(edgeright)
            rightcontact_y = int(min(np.argmax(np.abs(diff) > 5), np.argmax(diff[1:]*diff[:-1] < 0)))
            diff = np.diff(edgeleft)
            leftcontact_y = int(min(np.argmax(np.abs(diff) > 5), np.argmax(diff[1:]*diff[:-1] < 0)))
    else: # get contactpoints from jump of edge at baseline
        rightcontact_y = rightcontact.y
        leftcontact_y = leftcontact.y

    # polynomial fits of edges in larger region above baseline to get contact angles
    fitpointsleft = edgeleft[range(np.int(np.floor(leftcontact_y)),np.int(np.floor(leftcontact_y)-k),-1)]
    if any(fitpointsleft==0):
        fitpointsleft = np.delete(fitpointsleft,range(np.argmax(fitpointsleft==0),k))
    fitpointsright = edgeright[range(np.int(np.floor(rightcontact_y)),np.int(np.floor(rightcontact_y)-k),-1)]
    if any(fitpointsright==0):
        fitpointsright = np.delete(fitpointsright,range(np.argmax(fitpointsright==0),k))

    for i in range(2): # if order of polyfit is not optimal for contactangle, repeat fit with other order
        leftfit, res_left, _, _, _ = np.polyfit(range(0,fitpointsleft.shape[0]),fitpointsleft,PO_left, full = True) # polygonal fit for edge of drop
        leftvec = np.array([1,leftfit[PO_left-1]]) # tangential vector at contact point from linear coefficient of polygonal fit
        contactpointleft = leftfit[PO_left] # x-coordinate of contact point from absolute coefficient of polygonal fit
        thetal = np.arccos(np.dot(basevec,leftvec)/(np.sqrt(np.dot(basevec,basevec))*np.sqrt(np.dot(leftvec,leftvec))))*180/np.pi # contact angle (angle between tangential vector and baseline)
        if PO_left==2 and thetal>47 and thetal<=133:
            PO_left = 3
        elif PO_left==3 and (thetal<=43 or thetal>137):
            PO_left = 2
        else:
            break
    
    for i in range(2):
        rightfit, res_right, _, _, _  = np.polyfit(range(0,fitpointsright.shape[0]),fitpointsright,PO_right, full = True)
        rightvec = np.array([1,rightfit[PO_right-1]]) 
        contactpointright = rightfit[PO_right]
        thetar = 180-np.arccos(np.dot(basevec,rightvec)/(np.sqrt(np.dot(basevec,basevec))*np.sqrt(np.dot(rightvec,rightvec))))*180/np.pi
        if PO_right==2 and thetar>47 and thetar<=133:
            PO_right = 3
        elif PO_right==3 and (thetar<=43 or thetar>137):
            PO_right = 2
        else:
            break

    height = min(np.int(np.floor(leftcontact_y)),np.int(np.floor(rightcontact_y)))
    dropvolume = np.sum(np.pi*np.square((edgeright[0:height]-edgeleft[0:height])/2))
    # using cylindrical slice we calculate the remaining volume
    slantedbasediff=max(np.floor(leftcontact_y),np.floor(rightcontact_y))-min(np.floor(leftcontact_y),np.floor(rightcontact_y))
    # we assume that the radius is constant over the range of the slanted baseline, for small angles this is probably accurate, but for larger angles this can result in a significant error.
    baseradius = (edgeright[np.int(min(np.floor(leftcontact_y),np.floor(rightcontact_y)))]-edgeleft[np.int(min(np.floor(leftcontact_y),np.floor(rightcontact_y)))])/2
    dropvolume = dropvolume + 0.5*np.pi*np.square(baseradius)*slantedbasediff
    
    rightfitcurve=np.polyval(rightfit,np.arange(k))
    leftfitcurve=np.polyval(leftfit,np.arange(k))
    debug=np.array([leftfitcurve,leftcontact_y-np.arange(k),rightfitcurve,rightcontact_y-np.arange(k)])

    return contactpointleft, contactpointright, thetal, thetar, dropvolume, debug, PO_left, PO_right, leftcontact_y, rightcontact_y, res_right[0], res_left[0]
