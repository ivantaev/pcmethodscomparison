import numpy as np
from skimage.draw import polygon2mask
from skimage.measure import find_contours


def complete_contourline(c_x, c_y, xbound, ybound):
    """
    Parameters
    ----------
    c_x
    c_y
    xbound: tuple
        In unit of the index of the array (not the actual x corodinates)
    ybound
    Returns
    -------
    """

    xmin, xmax = xbound
    ymin, ymax = ybound

    x0, x1, y0, y1 = c_x[0], c_x[-1], c_y[0], c_y[-1]

    if (x0 == x1) and (y0 == y1):  # Closed loop
        case = 1
    elif (x0 == x1) or (y0 == y1):  # Either x or y doesn't end at the start
        case = 2
        c_x = np.append(c_x, x0)
        c_y = np.append(c_y, y0)  # Complete the loop

    else:  # Both x and y are not ending at the start
        case = 3

        if (x0 == xmin) or (x0 == xmax):  # if x starts at the edge
            c_x = np.append(c_x, x0)
            c_y = np.append(c_y, y1)

        elif (x1 == xmin) or (x1 == xmax):  # if x ends at the edge
            c_x = np.append(c_x, x1)
            c_y = np.append(c_y, y0)

        elif (y0 == ymin) or (y0 == ymax):  # if y starts at the edge
            c_x = np.append(c_x, x1)
            c_y = np.append(c_y, y0)

        elif (y1 == ymin) or (y1 == ymax):  # if y ends at the edge
            c_x = np.append(c_x, x0)
            c_y = np.append(c_y, y1)
        else:
            raise

        if np.all(np.array([c_x[0], c_y[0]]) == np.array([c_x[-1], c_y[-1]])) is False:  # if not close loop
            c_x = np.append(c_x, c_x[0])
            c_y = np.append(c_y, c_y[0])

    return c_x, c_y, case

def segment_fields(xx, yy, map, level=0.3):
    
    xbound = (0, xx.shape[1]-1)
    ybound = (0, yy.shape[0]-1)
    map = np.nan_to_num(map) #VI
    normmap = map / np.nanmax(map) #HY:max
    # Padding to make sure contour is always closed
    padded_normmap = np.zeros((normmap.shape[0]+10, normmap.shape[1]+10))
    padded_normmap[5:-5, 5:-5] = normmap
    contours = find_contours(padded_normmap, level=level) #HY:0.2
    c = 0
    for c_each in contours:
        c += 1
        c_rowi = c_each[:, 0] - 5  # y
        c_coli = c_each[:, 1] - 5  # x
        c_rowi, c_coli = np.clip(c_rowi, a_min=ybound[0], a_max=ybound[1]), np.clip(c_coli, a_min=xbound[0], a_max=xbound[1])
        c_coli, c_rowi, case = complete_contourline(c_coli, c_rowi, xbound=xbound, ybound=ybound)
        mask = polygon2mask(xx.shape, np.stack([c_rowi, c_coli]).T)  # Remember to transpose!
        c_rowi = np.around(c_rowi).astype(int)
        c_coli = np.around(c_coli).astype(int)
        c_x = xx[c_rowi, c_coli]
        c_y = yy[c_rowi, c_coli]
        xyval = np.stack([c_x, c_y]).T
        area = mask.sum()
        meanrate_in = np.mean(map[mask])
        if area < 2:
            continue
        
        yield area, meanrate_in, xyval, mask 
            