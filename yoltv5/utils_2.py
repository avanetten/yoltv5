import math
import multiprocessing
import os
import random
import time
import shutil
import sys
from subprocess import PIPE, STDOUT, Popen


from osgeo import gdal
from osgeo import ogr
from osgeo import osr

import cv2
import fiona
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
# import solaris.vector
import shapely
import skimage
import skimage.io
import skimage.transform
from affine import Affine
from matplotlib.collections import PolyCollection
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rtree.core import RTreeError
# from ..utils.geo import list_to_affine, _reduce_geom_precision
# from ..utils.core import _check_gdf_load, _check_crs, _check_rasterio_im_load
#from ..raster.image import get_geo_transform
from shapely.geometry import (MultiLineString, MultiPolygon, Polygon, box,
                              mapping, shape)
from shapely.wkt import loads
from statsmodels.stats.weightstats import DescrStatsW

from geo import affine_to_list, list_to_affine

###############################################################################
###############################################################################


# https://github.com/CosmiQ/solaris/blob/master/solaris/utils/geo.py
###############################################################################
def _reduce_geom_precision(geom, precision=2):
    geojson = mapping(geom)
    geojson['coordinates'] = np.round(np.array(geojson['coordinates']),
                                      precision)
    return shape(geojson)


# https://github.com/CosmiQ/solaris/blob/master/solaris/raster/image.py
###############################################################################
def get_geo_transform(raster_src):
    """Get the geotransform for a raster image source.
    Arguments
    ---------
    raster_src : str, :class:`rasterio.DatasetReader`, or `osgeo.gdal.Dataset`
        Path to a raster image with georeferencing data to apply to `geom`.
        Alternatively, an opened :class:`rasterio.Band` object or
        :class:`osgeo.gdal.Dataset` object can be provided. Required if not
        using `affine_obj`.
    Returns
    -------
    transform : :class:`affine.Affine`
        An affine transformation object to the image's location in its CRS.
    """

    if isinstance(raster_src, str):
        affine_obj = rasterio.open(raster_src).transform
    elif isinstance(raster_src, rasterio.DatasetReader):
        affine_obj = raster_src.transform
    elif isinstance(raster_src, gdal.Dataset):
        affine_obj = Affine.from_gdal(*raster_src.GetGeoTransform())

    return affine_obj
    
    
# https://github.com/CosmiQ/solaris/blob/master/solaris/vector/polygon.py
###############################################################################

def convert_poly_coords(geom, raster_src=None, affine_obj=None, inverse=False,
                        precision=None):
    """Georegister geometry objects currently in pixel coords or vice versa.
    Arguments
    ---------
    geom : :class:`shapely.geometry.shape` or str
        A :class:`shapely.geometry.shape`, or WKT string-formatted geometry
        object currently in pixel coordinates.
    raster_src : str, optional
        Path to a raster image with georeferencing data to apply to `geom`.
        Alternatively, an opened :class:`rasterio.Band` object or
        :class:`osgeo.gdal.Dataset` object can be provided. Required if not
        using `affine_obj`.
    affine_obj: list or :class:`affine.Affine`
        An affine transformation to apply to `geom` in the form of an
        ``[a, b, d, e, xoff, yoff]`` list or an :class:`affine.Affine` object.
        Required if not using `raster_src`.
    inverse : bool, optional
        If true, will perform the inverse affine transformation, going from
        geospatial coordinates to pixel coordinates.
    precision : int, optional
        Decimal precision for the polygon output. If not provided, rounding
        is skipped.
    Returns
    -------
    out_geom
        A geometry in the same format as the input with its coordinate system
        transformed to match the destination object.
    """

    if not raster_src and not affine_obj:
        raise ValueError("Either raster_src or affine_obj must be provided.")

    if raster_src is not None:
        affine_xform = get_geo_transform(raster_src)
    else:
        if isinstance(affine_obj, Affine):
            affine_xform = affine_obj
        else:
            # assume it's a list in either gdal or "standard" order
            # (list_to_affine checks which it is)
            if len(affine_obj) == 9:  # if it's straight from rasterio
                affine_obj = affine_obj[0:6]
            affine_xform = list_to_affine(affine_obj)

    if inverse:  # geo->px transform
        affine_xform = ~affine_xform

    if isinstance(geom, str):
        # get the polygon out of the wkt string
        g = shapely.wkt.loads(geom)
    elif isinstance(geom, shapely.geometry.base.BaseGeometry):
        g = geom
    else:
        raise TypeError('The provided geometry is not an accepted format. '
                        'This function can only accept WKT strings and '
                        'shapely geometries.')

    xformed_g = shapely.affinity.affine_transform(g, [affine_xform.a,
                                                      affine_xform.b,
                                                      affine_xform.d,
                                                      affine_xform.e,
                                                      affine_xform.xoff,
                                                      affine_xform.yoff])
    if isinstance(geom, str):
        # restore to wkt string format
        xformed_g = shapely.wkt.dumps(xformed_g)
    if precision is not None:
        xformed_g = _reduce_geom_precision(xformed_g, precision=precision)

    return xformed_g

###############################################################################
#http://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy   


###############################################################################
def create_mask(im_path, label_path, out_path_mask, burnValue=255):
    with fiona.open(label_path, "r") as annotation_collection:
        annotations = [feature["geometry"] for feature in annotation_collection]  
    with rasterio.open(im_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, annotations, 
            all_touched=False, invert=False, crop=False)
        out_meta = src.meta
        # clip
        out_image = burnValue * np.clip(out_image, 0, 1)
    htmp, wtmp = out_image.shape[1], out_image.shape[2]
    out_meta.update({"driver": "GTiff",
                 "height": htmp,
                 "width": wtmp,
                 "transform": out_transform})
    with rasterio.open(out_path_mask, "w", **out_meta) as dest:
        dest.write(out_image)
    

###############################################################################
def win_jitter(window_size, jitter_frac=0.1):
    '''get x and y jitter'''
    val = np.rint(jitter_frac * window_size)
    dx = np.random.randint(-val, val)
    dy = np.random.randint(-val, val)
    
    return dx, dy


###############################################################################
def get_window_geoms(df, window_size=416, jitter_frac=0.2, image_w=0, image_h=0,
                     geometry_col='geometry_poly_pixel', category_col='Category',
                     aug_count_dict=None,
                     verbose=False):
    '''Iterate through dataframe and get square window cutouts centered on each
    object, modulu some jitter,
    set category_col to None if none exists
    aug_count_dict is a dictionary detailing the number of augmentations to make,
        set to None to not augment'''
    
    geom_windows, geom_windows_aug = [], []
    len_df = len(df)
    i = 0
    for index, row in df.iterrows():
        cat = row[category_col]
        if verbose and category_col:
            print ("\n", i+1, "/", len_df, "category:", cat)
            # print ("\n", index, row['Category'])
        # get coords
        geom_pix = row[geometry_col]
        if type(geom_pix) == str:
            geom_pix = loads(geom_pix) 
            # print("  geom_pix:", geom_pix)
        #pix_coords = list(geom_pix.coords)
        bounds = geom_pix.bounds
        area = geom_pix.area
        (minx, miny, maxx, maxy) = bounds
        dx, dy = maxx-minx, maxy-miny
        if verbose:
            print ("  bounds:", bounds )
            print ("  dx, dy:", dx, dy )
            print ("  area:", area )
        
        # get centroid
        centroid = geom_pix.centroid
        #print "centroid:", centroid
        cx_tmp, cy_tmp = list(centroid.coords)[0]
        cx, cy = np.rint(cx_tmp), np.rint(cy_tmp)
        
        # get window coords, jitter, and shapely geometry for window
        # do this multiple times if augmentations are desired
        if aug_count_dict == None:
            n_wins = 1
        else:
            n_wins = 1 + aug_count_dict[cat]
        if verbose and category_col:
            print("  n_wins:", n_wins)
        
        for k in range(n_wins): 
            jx, jy = win_jitter(window_size, jitter_frac=jitter_frac)
            x0 = cx - window_size/2 + jx
            y0 = cy - window_size/2 + jy
            # ensure window does not extend outside larger image
            x0 = max(x0, 0)
            x0 = int(min(x0, image_w - window_size))
            y0 = max(y0, 0)
            y0 = int(min(y0, image_h - window_size))
            # set other side of square
            x1 = x0 + window_size
            y1 = y0 + window_size
            win_p1 = shapely.geometry.Point(x0, y0)
            win_p2 = shapely.geometry.Point(x1, y0)
            win_p3 = shapely.geometry.Point(x1, y1)
            win_p4 = shapely.geometry.Point(x0, y1)
            pointList = [win_p1, win_p2, win_p3, win_p4, win_p1]
            geom_window = shapely.geometry.Polygon([[p.x, p.y] for p in pointList])
            if verbose:
                print ("  geom_window.bounds", geom_window.bounds )
            # only append first to geom_window, others should be in windows_aug
            if k == 0:
                geom_windows.append(geom_window)
            else:
                geom_windows_aug.append(geom_window)
        i += 1

    return geom_windows, geom_windows_aug


###############################################################################
def tile_window_geoms(image_w, image_h, window_size=416, overlap_frac=0.2,
                     verbose=False):
    '''Create tiled square window cutouts for given image size
       Return a list of geometries for the windows
    '''
    
    sliceHeight = window_size
    sliceWidth = window_size
    dx = int((1. - overlap_frac) * sliceWidth)
    dy = int((1. - overlap_frac) * sliceHeight)
    
    n_ims = 0
    geom_windows = []
    for y0 in range(0, image_h, dy):#sliceHeight):
        for x0 in range(0, image_w, dx):#sliceWidth):
            n_ims += 1
            # ensure window does not extend outside larger image
            x0 = max(x0, 0)
            x0 = max(0, int(min(x0, image_w - sliceWidth)))
            y0 = max(y0, 0)
            y0 = max(0, int(min(y0, image_h - sliceHeight)))
            # set other side of square
            x1 = x0 + sliceWidth
            y1 = y0 + sliceHeight
            win_p1 = shapely.geometry.Point(x0, y0)
            win_p2 = shapely.geometry.Point(x1, y0)
            win_p3 = shapely.geometry.Point(x1, y1)
            win_p4 = shapely.geometry.Point(x0, y1)
            pointList = [win_p1, win_p2, win_p3, win_p4, win_p1]
            geom_window = shapely.geometry.Polygon([[p.x, p.y] for p in pointList])
            if verbose:
                print ("  geom_window.bounds", geom_window.bounds )
            # append
            geom_windows.append(geom_window)

    return geom_windows
    
    
###############################################################################
def get_objs_in_window(df_, geom_window, min_obj_frac=0.7, 
                       geometry_col='geometry_poly_pixel', category_col='Category',
                       use_box_geom=True, verbose=False):    
    '''Find all objects in the window
    if use_box_geom, turn the shapefile object geom into a bounding box
    return: [index_nest, cat_nest, x0_obj, y0_obj, x1_obj, y1_obj]'''
    
    (minx_win, miny_win, maxx_win, maxy_win) = geom_window.bounds
    if verbose:
        print ("geom_window.bounds:", geom_window.bounds)

    obj_list = []
    for index_nest, row_nest in df_.iterrows():
        cat_nest = row_nest[category_col]
        geom_pix_nest_tmp = row_nest[geometry_col]
        if type(geom_pix_nest_tmp) == str:
            geom_pix_nest_tmp = loads(geom_pix_nest_tmp)         
        # if use_box_geom, turn the shapefile object geom into a bounding box
        if use_box_geom:
            (x0, y0, x1, y1) = geom_pix_nest_tmp.bounds
            geom_pix_nest = shapely.geometry.box(x0, y0, x1, y1, ccw=True)
        else:
            geom_pix_nest = geom_pix_nest_tmp
        
        #pix_coords = list(geom_pix.coords)
        #bounds_nest = geom_pix_nest.bounds
        area_nest = geom_pix_nest.area
        
        # skip zero or negative areas
        if area_nest <= 0:
            continue
            
        # sometimes we get an invalid geometry, not sure why
        try:
            intersect_geom = geom_pix_nest.intersection(geom_window)
        except:
            # create a buffer around the exterior
            geom_pix_nest = geom_pix_nest.buffer(0)
            intersect_geom = geom_pix_nest.intersection(geom_window)
            print ("Had to update geom_pix_nest:", geom_pix_nest.bounds  )
            
        intersect_bounds = intersect_geom.bounds
        intersect_area = intersect_geom.area
        intersect_frac = intersect_area / area_nest
        
        # skip if object not in window, else add to window
        if intersect_frac < min_obj_frac:
            continue
        else:
            # get window coords
            (minx_nest, miny_nest, maxx_nest, maxy_nest) = intersect_bounds
            dx_nest, dy_nest = maxx_nest - minx_nest, maxy_nest - miny_nest
            x0_obj, y0_obj = minx_nest - minx_win, miny_nest - miny_win
            x1_obj, y1_obj = x0_obj + dx_nest, y0_obj + dy_nest
        
            x0_obj, y0_obj, x1_obj, y1_obj = np.rint(x0_obj), np.rint(y0_obj),\
                                             np.rint(x1_obj), np.rint(y1_obj)
            obj_list.append([index_nest, cat_nest, x0_obj, y0_obj, x1_obj, 
                             y1_obj])                                
            if verbose:
                print (" ", index_nest, "geom_obj.bounds:", geom_pix_nest.bounds )
                print ("  intesect area:", intersect_area )
                print ("  obj area:", area_nest )
                print ("  intersect_frac:", intersect_frac )
                print ("  intersect_bounds:", intersect_bounds )
                print ("  category:", cat_nest )
                
    return obj_list
    
    
###############################################################################
def get_image_window(im, window_geom):
    '''Get sub-window in image'''  
    
    bounds_int = [int(itmp) for itmp in window_geom.bounds]
    (minx_win, miny_win, maxx_win, maxy_win) = bounds_int
    window = im[miny_win:maxy_win, minx_win:maxx_win]
    return window


###############################################################################
def plot_obj_list(window, obj_list, color_dic, thickness=2,
                  show_plot=False, outfile=''):
    '''Plot the cutout, and the object bounds'''
        
    print ("window.shape:", window.shape )
    for row in obj_list:
        [index_nest, cat_nest, x0_obj, y0_obj, x1_obj, y1_obj] = row
        color = color_dic[cat_nest]
        cv2.rectangle(window, (int(x0_obj), int(y0_obj)), 
                      (int(x1_obj), int(y1_obj)), 
                      (color), thickness)    
        if show_plot:
            cv2.imshow(str(index_nest), window)
            cv2.waitKey(0)
    if outfile:
        cv2.imwrite(outfile, window)


###############################################################################
def plot_training_bboxes(label_folder, image_folder, ignore_augment=True,
                         figsize=(10, 10), color=(0, 0, 255), thickness=2,
                         max_plots=100, sample_label_vis_dir=None, ext='.png',
                         show_plot=False, specific_labels=[],
                         label_dic=[], output_width=60000, shuffle=True,
                         verbose=False):
    '''Plot bounding boxes for yolt
    specific_labels allows user to pass in labels of interest'''

    out_suff = ''  # '_vis'

    if sample_label_vis_dir and not os.path.exists(sample_label_vis_dir):
        os.mkdir(sample_label_vis_dir)

    # boats, boats_harbor, airplanes, airports (blue, green, red, orange)
    # remember opencv uses bgr, not rgb
    colors = 40*[(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 140, 255),
              (0, 255, 125), (125, 125, 125), (140, 200, 0), (50, 200, 255),
              (0, 102, 0), (255, 0, 127), (51, 0, 105), (153, 0, 0), 
              (0, 128, 250), (255, 255, 100), (127, 0, 255), (153, 76, 0)]
    #colorsmap =  plt.cm.gist_rainbow
    #colors = [colormap(i) for i in np.linspace(0, 0.9, len(archs))]

    if verbose:
        print("colors:", colors)

    try:
        cv2.destroyAllWindows()
    except:
        pass
    
    i = 0

    if len(specific_labels) == 0:
        label_list = os.listdir(label_folder)
        # shuffle?
        if shuffle:
            random.shuffle(label_list)

    else:
        label_list = specific_labels

    for label_file in label_list:

        if ignore_augment:
            if (label_file == '.DS_Store') or (label_file.endswith(('_lr.txt', '_ud.txt', '_lrud.txt'))):
                continue
        # else:
        #     if (label_file == '.DS_Store'):
        #         continue

        if i >= max_plots:
            # print "i, max_plots:", i, max_plots
            return
        else:
            i += 1

        if verbose:
            print(i, "/", max_plots)
            print("  label_file:", label_file)

        # get image
        # root = label_file.split('.')[0]
        root = label_file[:-4]
        im_loc = os.path.join(image_folder,  root + ext)
        label_loc = os.path.join(label_folder, label_file)
        if verbose:
            print(" root:", root)
            print("  label loc:", label_loc)
            print("  image loc:", im_loc)
        image0 = cv2.imread(im_loc, 1)
        height, width = image0.shape[:2]
        # resize output file
        if output_width < width:
            height_mult = 1.*height / width
            output_height = int(height_mult * output_width)
            outshape = (output_width, output_height)
            image = cv2.resize(image0, outshape)
        else:
            image = image0

        height, width = image.shape[:2]
        shape = (width, height)
        if verbose:
            print("im.shape:", image.shape)

        # start plot (mpl)
        #fig, ax = plt.subplots(figsize=figsize)
        #img_mpl = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # ax.imshow(img_mpl)
        # just opencv
        img_mpl = image

        # get and plot labels
        # z = pd.read_csv(label_folder + label_file, sep = ' ', names=['cat', 'x', 'y', 'w', 'h'])
        z = pd.read_csv(label_loc, sep=' ', names=['cat', 'x', 'y', 'w', 'h'])
        # print "z", z.values
        for yolt_box in z.values:
            cat_int = int(yolt_box[0])
            color = colors[cat_int]
            yb = yolt_box[1:]
            box0 = convert_reverse(shape, yb)
            # convert to int
            box1 = [int(round(b, 2)) for b in box0]
            [xmin, xmax, ymin, ymax] = box1
            # plot
            cv2.rectangle(img_mpl, (xmin, ymin),
                          (xmax, ymax), (color), thickness)

        # add border
        if label_dic:

            # https://codeyarns.files.wordpress.com/2015/03/20150311_opencv_fonts.png
            font = cv2.FONT_HERSHEY_TRIPLEX  # FONT_HERSHEY_SIMPLEX #_SIMPLEX _TRIPLEX
            font_size = 0.25
            label_font_width = 1
            #text_offset = [3, 10]
            if len(label_dic.items()) <= 10:
                ydiff = 35
            else:
                ydiff = 22

            # add border
            # http://docs.opencv.org/3.1.0/d3/df2/tutorial_py_basic_ops.html
            # top, bottom, left, right - border width in number of pixels in corresponding directions
            border = (0, 0, 0, 200)
            border_color = (255, 255, 255)
            label_font_width = 1
            img_mpl = cv2.copyMakeBorder(img_mpl, border[0], border[1], border[2], border[3],
                                         cv2.BORDER_CONSTANT, value=border_color)
            # add legend
            xpos = img_mpl.shape[1] - border[3] + 15
            # for itmp, k in enumerate(sorted(label_dic.keys())):
            # for itmp, (k, value) in enumerate(sorted(label_dic.items(), key=operator.itemgetter(1))):
            for itmp, (k, value) in enumerate(sorted(label_dic.items(), key=lambda item: item[1])):
                labelt = label_dic[k]
                colort = colors[k]
                #labelt, colort = label_dic[k]
                text = '- ' + labelt  # str(k) + ': ' + labelt
                ypos = ydiff + (itmp) * ydiff
                # cv2.putText(img_mpl, text, (int(xpos), int(ypos)), font, 1.5*font_size, colort, label_font_width, cv2.CV_AA)#, cv2.LINE_AA)
                cv2.putText(img_mpl, text, (int(xpos), int(ypos)), font, 1.5 *
                            # font_size, colort, label_font_width, cv2.CV_AA)  # cv2.LINE_AA)
                            font_size, colort, label_font_width, cv2.LINE_AA)

            # legend box
            cv2.rectangle(img_mpl, (xpos-5, 2*border[0]), (img_mpl.shape[1]-10, ypos+int(
                0.75*ydiff)), (0, 0, 0), label_font_width)

            # title
            # title = figname.split('/')[-1].split('_')[0] + ':  Plot Threshold = ' + str(plot_thresh) # + ': thresh=' + str(plot_thresh)
            #title_pos = (border[0], int(border[0]*0.66))
            # cv2.putText(img_mpl, title, title_pos, font, 1.7*font_size, (0,0,0), label_font_width, cv2.CV_AA)#, cv2.LINE_AA)
            # cv2.putText(img_mpl, title, title_pos, font, 1.7*font_size, (0,0,0), label_font_width,  cv2.CV_AA)#cv2.LINE_AA)

        if show_plot:
            cv2.imshow(root, img_mpl)
            cv2.waitKey(0)

        if sample_label_vis_dir:
            fout = os.path.join(sample_label_vis_dir,  root + out_suff + ext)
            cv2.imwrite(fout, img_mpl)

    return


###############################################################################
def augment_training_data(label_folder, image_folder,
                          label_folder_out='', image_folder_out='',
                          hsv_range=[0.5, 1.5],
                          skip_hsv_transform=True, ext='.jpg'):
    '''
    From yolt_data_prep_funcs.py
    Rotate data to augment training sizeo 
    darknet c functions already to HSV transform, and left-right swap, so
    skip those transforms
    Image augmentation occurs in data.c load_data_detection()'''

    if len(label_folder_out) == 0:
        label_folder_out = label_folder
    if len(image_folder_out) == 0:
        image_folder_out = image_folder

    hsv_diff = hsv_range[1] - hsv_range[0]
    im_l_out = []
    for label_file in os.listdir(label_folder):

        # don't augment the already agumented data
        if (label_file == '.DS_Store') or \
                (label_file.endswith(('_lr.txt', '_ud.txt', '_lrud.txt', '_rot90.txt', '_rot180.txt', '_rot270.txt'))):
            continue

        # get image
        print("image loc:", label_file)
        root = label_file.split('.')[0]
        im_loc = os.path.join(image_folder, root + ext)

        #image = skimage.io.imread(f, as_grey=True)
        image = cv2.imread(im_loc, 1)

        # randoly scale in hsv space, create a list of images
        if skip_hsv_transform:
            img_hsv = image
        else:
            try:
                img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            except:
                continue

        img_out_l = []
        np.random.seed(42)

        # three mirrorings
        if skip_hsv_transform:
            img_out_l = 6*[image]

        else:
            for i in range(6):
                im_tmp = img_hsv.copy()
                # alter values for each of 2 bands (hue and saturation)
                # for j in range(2):
                #    rand = hsv_range[0] + hsv_diff*np.random.random()  # between 0,5 and 1.5
                #    z0 = (im_tmp[:,:,j]*rand).astype(int)
                #    im_tmp[:,:,j] = z0
               # alter values for each of 3 bands (hue and saturation, value
                for j in range(3):
                    # set 'value' range somewhat smaller
                    if j == 2:
                        rand = 0.7 + 0.6*np.random.random()
                    else:
                        rand = hsv_range[0] + hsv_diff * \
                            np.random.random()  # between 0,5 and 1.5
                    z0 = (im_tmp[:, :, j]*rand).astype(int)
                    z0[z0 > 255] = 255
                    im_tmp[:, :, j] = z0

                # convert back to bgr and add to list of ims
                img_out_l.append(cv2.cvtColor(im_tmp, cv2.COLOR_HSV2BGR))

        # print "image.shape", image.shape
        # reflect or flip image left to right (skip since yolo.c does this?)
        image_lr = np.fliplr(img_out_l[0])  # (image)
        image_ud = np.flipud(img_out_l[1])  # (image)
        image_lrud = np.fliplr(np.flipud(img_out_l[2]))  # (image_ud)

        #cv2.imshow("in", image)
        #cv2.imshow("lr", image_lr)
        #cv2.imshow("ud", image_ud)
        #cv2.imshow("udlr", image_udlr)

        image_rot90 = np.rot90(img_out_l[3])
        image_rot180 = np.rot90(np.rot90(img_out_l[4]))
        image_rot270 = np.rot90(np.rot90(np.rot90(img_out_l[5])))

        # flip coords of bounding boxes too...
        # boxes have format: (x,y,w,h)
        z = pd.read_csv(os.path.join(label_folder, label_file),
                        sep=' ', names=['x', 'y', 'w', 'h'])

        # left right flip
        lr_out = z.copy()
        lr_out['x'] = 1. - z['x']

        # left right flip
        ud_out = z.copy()
        ud_out['y'] = 1. - z['y']

        # left right, up down, flip
        lrud_out = z.copy()
        lrud_out['x'] = 1. - z['x']
        lrud_out['y'] = 1. - z['y']

        ##################
        # rotate bounding boxes X degrees
        origin = [0.5, 0.5]
        point = [z['x'], z['y']]

        # 90 degrees
        angle = -1*np.pi/2
        xn, yn = rotate(origin, point, angle)
        rot_out90 = z.copy()
        rot_out90['x'] = xn
        rot_out90['y'] = yn
        rot_out90['h'] = z['w']
        rot_out90['w'] = z['h']

        # 180 degrees (same as lrud)
        angle = -1*np.pi
        xn, yn = rotate(origin, point, angle)
        rot_out180 = z.copy()
        rot_out180['x'] = xn
        rot_out180['y'] = yn

        # 270 degrees
        angle = -3*np.pi/2
        xn, yn = rotate(origin, point, angle)
        rot_out270 = z.copy()
        rot_out270['x'] = xn
        rot_out270['y'] = yn
        rot_out270['h'] = z['w']
        rot_out270['w'] = z['h']
        ##################

        # print to files, add to list
        im_l_out.append(im_loc)

#        # reflect or flip image left to right (skip since yolo.c does this?)
#        imout_lr = image_folder + root + '_lr.jpg'
#        labout_lr = label_folder + root + '_lr.txt'
#        cv2.imwrite(imout_lr, image_lr)
#        lr_out.to_csv(labout_lr, sep=' ', header=False)
#        #im_l_out.append(imout_lr)

        # flip vertically or rotate 180 randomly
        if bool(random.getrandbits(1)):
            # flip vertically
            imout_ud = os.path.join(image_folder_out, root + '_ud' + ext)
            labout_ud = os.path.join(label_folder_out, root + '_ud.txt')
            cv2.imwrite(imout_ud, image_ud)
            ud_out.to_csv(labout_ud, sep=' ', header=False)
            im_l_out.append(imout_ud)
        else:
            im180_path = os.path.join(image_folder_out, root + '_rot180' + ext)
            cv2.imwrite(os.path.join(im180_path), image_rot180)
            rot_out180.to_csv(os.path.join(label_folder_out,
                                           root + '_rot180.txt'), sep=' ', header=False)
            im_l_out.append(im180_path)


#        # lrud flip, same as rot180
#        #  skip lrud flip because yolo does this sometimes
#        imout_lrud = image_folder + root + '_lrud.jpg'
#        labout_lrud = label_folder + root + '_lrud.txt'
#        cv2.imwrite(imout_lrud, image_lrud)
#        lrud_out.to_csv(labout_lrud, sep=' ', header=False)
#        #im_l_out.append(imout_lrud)

        # same as _lrud
        #im180 = image_folder + root + '_rot180.jpg'
        #cv2.imwrite(image_folder + root + '_rot180.jpg', image_rot180)
        #rot_out180.to_csv(label_folder + root + '_rot180.txt', sep=' ', header=False)
        # im_l_out.append(im180)

        # rotate 90 degrees or 270 randomly
        if bool(random.getrandbits(1)):
            im90_path = os.path.join(image_folder_out, root + '_rot90' + ext)
            #lab90 = label_folder + root + '_rot90.txt'
            cv2.imwrite(im90_path, image_rot90)
            rot_out90.to_csv(os.path.join(label_folder_out,
                                          root + '_rot90.txt'), sep=' ', header=False)
            im_l_out.append(im90_path)

        else:
            # rotate 270 degrees ()
            im270_path = os.path.join(image_folder_out, root + '_rot270' + ext)
            cv2.imwrite(im270_path, image_rot270)
            rot_out270.to_csv(os.path.join(label_folder_out,
                                           root + '_rot270.txt'), sep=' ', header=False)
            im_l_out.append(im270_path)

    return im_l_out


###############################################################################
def rm_augment_training_data(label_folder, image_folder, tmp_dir):
    '''Remove previusly created augmented data since it's done in yolt.c and 
    need not be doubly augmented'''

    # mv augmented labels
    for label_file in os.listdir(label_folder):

        if (label_file.endswith(('_lr.txt', '_ud.txt', '_lrud.txt', '_rot90.txt', '_rot180.txt', '_rot270.txt'))):

            try:
                os.mkdir(tmp_dir)
            except:
                print("")

            # mv files to tmp_dir
            print("label_file", label_file)
            #shutil.move(label_file, tmp_dir)
            # overwrite:
            shutil.move(os.path.join(label_folder, label_file),
                        os.path.join(tmp_dir, label_file))

            # just run images separately below to make sure we get errthing
            # get image
            # print "image loc:", label_file
            #root = label_file.split('.')[0]
            #im_loc = image_folder + root + '.jpg'
            # mv files to tmp_dir
            #shutil.move(im_loc, tmp_dir)

    # mv augmented images
    for image_file in os.listdir(image_folder):

        if (image_file.endswith(('_lr.jpg', '_ud.jpg', '_lrud.jpg', '_rot90.jpg', '_rot180.jpg', '_rot270.jpg'))):

            try:
                os.mkdir(tmp_dir)
            except:
                print("")

            # mv files to tmp_dir
            #shutil.move(image_file, tmp_dir)
            # overwrite
            shutil.move(os.path.join(image_folder, image_file),
                        os.path.join(tmp_dir, image_file))

    return


###############################################################################
def yolt_from_df(im_path, df_polys, 
                    window_size=512, 
                    jitter_frac=0.1,
                    min_obj_frac=0.7,
                    max_obj_count=1000,
                    geometry_col='geometry', 
                    category_col='make_id',
                    image_fname_col='image_fname',
                    outdir_ims=None, 
                    outdir_labels=None,
                    outdir_yolt_plots=None,
                    outdir_ims_aug=None, 
                    outdir_labels_aug=None,
                    outdir_yolt_plots_aug=None,                    
                    aug_count_dict=None,
                    yolt_image_ext='.jpg',
                    max_plots=10,
                    flip_vert_prob=0,
                    verbose=True, super_verbose=False):
    '''
    Extract yolt cutouts and labels from a singe image.
    df_polys is created from: DA_Dataset/gj_to_px_bboxes.ipynb, which converts geojsons to csvs with columns:
    image_fname, cat_id, loc_id, location, original_make, make, da_make_id, pnp_id, geometry 
    aug_count_dict is a dictionary detailing the number of augmentations to make,
        set to none to not augment
    flip_vert_prob is the probabilty of randomly flipping each extracted chip vertically (set to 0 to skip)
    '''
    
    # ensure image exists
    if not os.path.exists(im_path):
        print (" Image file {} DNE...".format(im_path) )
        return
    else:
        im = skimage.io.imread(im_path)
        im_name = os.path.basename(im_path)
        im_root = im_name.split('.')[0]
        image_h, image_w = im.shape[:2]
    # filter dataframe 
    df_im_tmp = df_polys[df_polys[image_fname_col] == im_name]
    
    if verbose:
        print("image", im_name)
        print("  im.shape:", im.shape)
        print("  object count:", df_im_tmp[category_col].value_counts().to_dict())
        # print("  len df_im:", len(df_im_tmp))
    
    # get window cutouts centered at each object
    window_geoms, window_geoms_aug = get_window_geoms(df_im_tmp, 
                                    window_size=window_size, 
                                    image_w=image_w, image_h=image_h,
                                    jitter_frac=jitter_frac, 
                                    geometry_col=geometry_col, 
                                    category_col=category_col,
                                    aug_count_dict=aug_count_dict,
                                    verbose=super_verbose)

    # set up dicts for counting objects
    idx_count_dic = {}
    for idx_tmp in df_im_tmp.index:
        idx_count_dic[idx_tmp] = 0
    idx_count_tot_dic = {}
    for idx_tmp in df_im_tmp.index:
        idx_count_tot_dic[idx_tmp] = 0  
        
    # get objects in each window
    win_iter = 0
    for i, window_geom in enumerate(window_geoms):

        (minx_win, miny_win, maxx_win, maxy_win) = window_geom.bounds
        
        # get window
        window = get_image_window(im, window_geom)
        h, w = window.shape[:2]
        if (h==0) or (w==0):
            continue

        # get objects in window
        obj_list = get_objs_in_window(df_im_tmp, window_geom, 
                                      min_obj_frac=min_obj_frac,
                                      geometry_col=geometry_col, 
                                      category_col=category_col,
                                      use_box_geom=True,
                                      verbose=super_verbose)  
        if super_verbose:
            print ("\nWindow geom:", window_geom )
            print ("  window shape:", window.shape )
            print ("  obj_list:", obj_list )
    
        if len(obj_list) > 0 :
            
            # update idx_count_tot_dic
            idxs_list = [z[0] for z in obj_list]
            for idx_tmp in idxs_list:
                idx_count_tot_dic[idx_tmp] += 1
                
            # Check idx count dic.  If an object has appeared too frequently,
            #   skip the window
            excess = False
            for idx_tmp in idxs_list:
                if idx_count_dic[idx_tmp] >= max_obj_count:
                    print ("Index", idx_tmp, "seen too frequently, skipping..." )
                    excess = True
                    break
            if excess:
                continue
            
            # create and save yolt images and labels
            outroot =  im_root + '__' + 'x0_' + str(int(minx_win)) + '_y0_' \
                                          + str(int(miny_win)) + '_dxdy_' \
                                          + str(int(window_size))
            
            if not (outdir_ims and outdir_labels):
                return
    
            # get yolt labels
            #if verbose:
            #    print ("  Creating yolt labels..."
            yolt_coords = []
            for row in obj_list:
                [index_nest, cat_nest, x0, y0, x1, y1] = row
                yolt_row = [cat_nest] + list(convert((w,h), [x0,x1,y0,y1]))
                # cat_idx = cat_idx_dic[cat_nest]
                # yolt_row = [cat_idx, cat_nest] + list(convert.convert((w,h), [x0,x1,y0,y1]))
                yolt_coords.append(yolt_row)
            if super_verbose:
                print ("   yolt_coords:", yolt_coords )
            
            # if desired, and randomly selected, flip vertically 
            use_flip = False
            flip_rand_val = random.uniform(0, 1)
            if (flip_vert_prob > 0) and (flip_rand_val < flip_vert_prob):
                use_flip = True
                # flip yolt coords ((cat, x, y, w, h))
                yolt_coords_flip = []
                for yc in yolt_coords:
                    yolt_coords_flip.append([yc[0], yc[1], 1.0 - yc[2], yc[3], yc[4]])
                yolt_coords = yolt_coords_flip
                # flip image
                window = np.flipud(window)
                flip_suff = '_ud'
            else:
                flip_suff = ''
            
            # set outfiles
            image_outfile = os.path.join(outdir_ims, outroot + flip_suff + yolt_image_ext)
            label_outfile = os.path.join(outdir_labels, outroot +flip_suff + '.txt')
            
            # save image
            skimage.io.imsave(image_outfile, window)
            # cv2.imwrite(image_outfile, window)

            # save labels
            txt_outfile = open(label_outfile, "w")
            for j, yolt_row in enumerate(yolt_coords):
                cls_id = yolt_row[0]
                bb = yolt_row[1:]
                # bb = yolt_row[2:]
                outstring = str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n'
                # print("outstring:", outstring
                txt_outfile.write(outstring)
            txt_outfile.close()             

            # update idx_count_dic
            for idx_tmp in idxs_list:
                idx_count_dic[idx_tmp] += 1
            # update win_iter
            win_iter += 1
                
    # make sure labels and images are created correctly
    print ("\nPlotting yolt training bounding boxes..." )
    print ("  outdir_labels:", outdir_labels )
    print ("  outdir_ims:", outdir_ims )
    plot_training_bboxes(outdir_labels, outdir_ims, 
                 ignore_augment=True,
                 figsize=(10,10), color=(0,0,255), thickness=2, 
                 max_plots=max_plots, sample_label_vis_dir=outdir_yolt_plots,
                 ext=yolt_image_ext, show_plot=False, 
                 specific_labels=[], label_dic=[], output_width=500,
                 shuffle=True, verbose=super_verbose) 


    # now handle augmented windows
    # rotate the windows.  yolo already does lr flips, so we can skip any of that type of aug
    if len(window_geoms_aug) > 0:
        
        # get objects in each window
        win_iter = 0
        for i, window_geom in enumerate(window_geoms_aug):
            (minx_win, miny_win, maxx_win, maxy_win) = window_geom.bounds
            # get window
            window = get_image_window(im, window_geom)
            h, w = window.shape[:2]
            if (h==0) or (w==0):
                continue
            # get objects in window
            obj_list = get_objs_in_window(df_im_tmp, window_geom, 
                                          min_obj_frac=min_obj_frac,
                                          geometry_col=geometry_col, 
                                          category_col=category_col,
                                          use_box_geom=True,
                                          verbose=super_verbose)  
            if super_verbose:
                print ("\nWindow geom:", window_geom )
                print ("  window shape:", window.shape )
                print ("  obj_list:", obj_list )
    
            # skip if empty, obviously
            if len(obj_list) == 0:
                continue
            
            # elif not empty, rotate, see augment_training_data() for code
            # just apply a rotation to every third (
            # (l-r flipping is standard within yolo, so we can skip that augmentation)
            else:
                # get yolt labels
                yolt_coords = []
                for row in obj_list:
                    [index_nest, cat_nest, x0, y0, x1, y1] = row
                    yolt_row = [cat_nest] + list(convert((w,h), [x0,x1,y0,y1]))
                    # cat_idx = cat_idx_dic[cat_nest]
                    # yolt_row = [cat_idx, cat_nest] + list(convert.convert((w,h), [x0,x1,y0,y1]))
                    yolt_coords.append(yolt_row)
                df_yc = pd.DataFrame(yolt_coords, columns=['cat', 'x', 'y', 'w', 'h'])
        
                # first, flip vertically
                n_options = 3
                if (i % n_options) == 0:
                    suff = '_ud'
                    win_out = np.flipud(window)
                    df_out = df_yc.copy()
                    df_out['y'] = 1. - df_yc['y']  
                # second, rotate 90
                elif ((i-1) % n_options) == 0:
                    suff = '_rot90'
                    win_out = np.rot90(window)
                    # 90 degrees
                    origin = [0.5, 0.5]
                    point = [df_yc['x'], df_yc['y']]
                    angle = -1*np.pi/2
                    xn, yn = rotate(origin, point, angle)
                    df_out = df_yc.copy()
                    df_out['x'] = xn
                    df_out['y'] = yn
                    df_out['h'] = df_yc['w']
                    df_out['w'] = df_yc['h']   
                # third, rotate 270
                elif ((i-2) % n_options) == 0:
                    suff = '_rot270'
                    win_out = np.rot90(window, 3)
                    # 90 degrees
                    origin = [0.5, 0.5]
                    point = [df_yc['x'], df_yc['y']]
                    angle = -3*np.pi/2
                    xn, yn = rotate(origin, point, angle)
                    df_out = df_yc.copy()
                    df_out['x'] = xn
                    df_out['y'] = yn
                    df_out['h'] = df_yc['w']
                    df_out['w'] = df_yc['h']
                    # # fourth, rotate 180
                    # elif ((i-3) % n_options) == 0:
                    #     suff = '_rot180'
                    #     win_out = np.rot180(window, 3)
                    #     # 90 degrees
                    #     origin = [0.5, 0.5]
                    #     point = [df_yc['x'], df_yc['y']]
                    #     angle = np.pi
                    #     xn, yn = rotate(origin, point, angle)
                    #     df_out = df_yc.copy()
                    #     df_out['x'] = xn
                    #     df_out['y'] = yn
                    #     df_out['h'] = df_yc['h']
                    #     df_out['w'] = df_yc['w']
                    # # fifth, flip rotate 180?

                # create and save yolt images and labels
                outroot =  im_root + '__' + 'x0_' + str(int(minx_win)) + '_y0_' \
                                              + str(int(miny_win)) + '_dxdy_' \
                                              + str(int(window_size)) + suff
                # paths                                 
                image_outfile = os.path.join(outdir_ims_aug, outroot + yolt_image_ext)
                label_outfile = os.path.join(outdir_labels_aug, outroot + '.txt')
                # save
                skimage.io.imsave(image_outfile, win_out)
                df_out.to_csv(label_outfile, sep=' ', header=False, index=False)
                
        # make sure labels and images are created correctly
        print ("\nPlotting yolt training bounding boxes..." )
        print ("  outdir_labels:", outdir_labels )
        print ("  outdir_ims:", outdir_ims )
        plot_training_bboxes(outdir_labels_aug, outdir_ims_aug, 
                     ignore_augment=False,
                     figsize=(10,10), color=(0,0,255), thickness=2, 
                     max_plots=max_plots, sample_label_vis_dir=outdir_yolt_plots_aug,
                     ext=yolt_image_ext, show_plot=False, 
                     specific_labels=[], label_dic=[], output_width=500,
                     shuffle=True, verbose=super_verbose) 
    
    return


###############################################################################
def yolt_from_visdrone(im_path, label_path, 
                    geometry_col='geometry', 
                    category_col='object_category',
                    window_size=416, 
                    overlap_frac=0.1,
                    min_obj_frac=0.7,
                    max_obj_count=10000,
                    outdir_ims=None, 
                    outdir_labels=None,
                    outdir_yolt_plots=None,
                    yolt_image_ext='.jpg',
                    # min_bbox_extent=24,
                    min_bbox_area_pix=256,
                    max_plots=10,
                    label_sep=',',
                    label_col_names=['bbox_left','bbox_top','bbox_width','bbox_height','score','object_category','truncation','occlusion'],
                    input_label_name_dict={0:'ignored', 1:'pedestrian', 2:'people', 3:'bicycle', 4:'car', 
                                     5:'van', 6:'truck', 7:'tricycle', 8:'awning-tricycle', 9:'bus', 10:'motor', 11:'others'},
                    label_int_conv_dict={1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10},
                    max_occlusion_int=1,
                    category_prime_col='category_int',
                    category_prime_col_str='category_str',
                    overwrite=False, verbose=True, super_verbose=False):
    '''
    Extract yolt cutouts and labels from a singe image/label pair in 
                    the VisDrone dataset (https://github.com/VisDrone/VisDrone2018-DET-toolkit).
    <object_category>    The object category indicates the type of annotated object, (i.e., ignored regions(0), pedestrian(1), 
                         people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10), 
                         others(11))
    <occlusion>	     The score in the DETECTION file should be set to the constant -1.
                         The score in the GROUNDTRUTH file indicates the fraction of objects being occluded (i.e., no occlusion = 0 
                         (occlusion ratio 0%), partial occlusion = 1 (occlusion ratio 1% ~ 50%), and heavy occlusion = 2 
                         (occlusion ratio 50% ~ 100%)).data
    
    label_int_conv_dict is a dict to convert the raw labels (from input_label_name_dict) to the desired output integers labels.
    skip category 0 because these are ignored regions    
    min_bbox_area_pix is the minimum area (in pixels) of a bounding box - any smaller and we drop it                
    (unused): min_bbox_extent is the minimum size of a bounding box - any smaller and we drop it.                        
    '''
    
    # define label_str_dict
    label_str_dict={}
    for k,v in label_int_conv_dict.items():
        label_str_dict[v] = input_label_name_dict[k]
    
    # ensure image exists
    if not os.path.exists(im_path):
        print (" Image file {} DNE...".format(im_path) )
        return
    else:
        # read in image
        im = skimage.io.imread(im_path)
        im_name = os.path.basename(im_path)
        im_root = im_name.split('.')[0]
        image_h, image_w = im.shape[:2]

    # read in labels, get coords
    df_label = pd.read_csv(label_path, sep=label_sep, names=label_col_names, header=None)
    df_label['xmin'] = df_label['bbox_left']
    df_label['xmax'] = df_label['bbox_left'] + df_label['bbox_width']
    df_label['ymin'] = df_label['bbox_top']
    df_label['ymax'] = df_label['bbox_top'] + df_label['bbox_height']
    # xext = df_label['xmax'] - df_label['xmin']
    # yext = df_label['ymax'] - df_label['ymin']
    # df_label['extent'] = [min(a,b) for (a,b) in zip(xext, yext)]
    
    # ceate geometry, area column
    geom_list_tmp = []
    area_list_tmp = []
    for idx_tmp, row_tmp in df_label.iterrows():
        geom_tmp = shapely.geometry.box(row_tmp['xmin'], row_tmp['ymin'], row_tmp['xmax'], row_tmp['ymax'], ccw=True)
        geom_list_tmp.append(geom_tmp)
        area_list_tmp.append(geom_tmp.area)
    df_label[geometry_col] = geom_list_tmp
    df_label['area'] = area_list_tmp
    # get label names?
    # df_label[category_str_col] = [input_label_name_dict[z] for z in df_label[category_col].values]
        
    #########################
    # now address category 0 = 'ignored'
    # let's black out the portion of the image that's supposed to be ignored, also remove label
    df_ig = df_label[df_label[category_col] == 0]
    for idx_tmp, row_ig in df_ig.iterrows():
        # set image to 0 in these regions
        im[row_ig['ymin']:row_ig['ymax'], row_ig['xmin']:row_ig['xmax']] = 0
    # remove all cats where category == 0!
    df_label_filt = df_label[df_label[category_col] > 0]

    # now let's shift labels (actually, do this later now)
    # df_label[category_prime_col] = df_label[category_col] - 1
    # make new dict with subtracted keys?
    # actually, we import label_dict_prime now...
    # label_name_dict_prime = {}
    # for k,v in input_label_name_dict.items():
    #     if k == 0:
    #        continue
    #    else:
    #        label_name_dict_prime[k-1] = v    
    #########################
    
    # remove rows with labels not in label_name_dict_prime
    good_cats_list = list(label_int_conv_dict.keys())
    df_label_filt = df_label_filt[df_label_filt[category_col].isin(good_cats_list)]
    
    # filter out bboxes that are too small
    df_label_filt = df_label_filt[df_label_filt['area'] >= min_bbox_area_pix]
    # df_label_filt = df_label_filt[df_label_filt['extent'] >= min_bbox_extent]

    # filter out bboxes that are too occluded
    df_label_filt = df_label_filt[df_label_filt['occlusion'] <= max_occlusion_int]
        
    # now let's create output labels
    df_label_filt[category_prime_col] = [label_int_conv_dict[z] for z in df_label_filt[category_col].values]
    df_label_filt[category_prime_col_str] = [label_str_dict[z] for z in df_label_filt[category_prime_col].values]
    
    # get window geoms
    window_geoms = tile_window_geoms(image_w, image_h, window_size=window_size, 
                        overlap_frac=overlap_frac, verbose=super_verbose)

    # set up dicts for counting objects
    idx_count_dic = {}
    for idx_tmp in df_label_filt.index:
        idx_count_dic[idx_tmp] = 0
    idx_count_tot_dic = {}
    for idx_tmp in df_label_filt.index:
        idx_count_tot_dic[idx_tmp] = 0  
        
    # get objects in each window
    win_iter = 0
    for i, window_geom in enumerate(window_geoms):

        (minx_win, miny_win, maxx_win, maxy_win) = window_geom.bounds
        
        # name root
        outroot =  im_root + '__' + 'x0_' + str(int(minx_win)) + '_y0_' \
                                      + str(int(miny_win)) + '_dxdy_' \
                                      + str(int(window_size))          
        flip_suff = ''
        # set outfiles
        image_outfile = os.path.join(outdir_ims, outroot + flip_suff + yolt_image_ext)
        label_outfile = os.path.join(outdir_labels, outroot +flip_suff + '.txt')
        # skip if we don't want to overwrite
        if not overwrite and os.path.exists(image_outfile):
            continue
        
        # get window
        window = get_image_window(im, window_geom)
        h, w = window.shape[:2]
        if (h==0) or (w==0):
            continue

        # get objects in window
        obj_list = get_objs_in_window(df_label_filt, window_geom, 
                                      min_obj_frac=min_obj_frac,
                                      geometry_col=geometry_col, 
                                      category_col=category_prime_col,
                                      use_box_geom=True,
                                      verbose=super_verbose)  
        if super_verbose:
            print ("\nWindow geom:", window_geom )
            print ("  window shape:", window.shape )
            print ("  obj_list:", obj_list )
    
        if len(obj_list) > 0 :
            
            # update idx_count_tot_dic
            idxs_list = [z[0] for z in obj_list]
            for idx_tmp in idxs_list:
                idx_count_tot_dic[idx_tmp] += 1
                
            # Check idx count dic.  If an object has appeared too frequently,
            #   skip the window
            excess = False
            for idx_tmp in idxs_list:
                if idx_count_dic[idx_tmp] >= max_obj_count:
                    print ("Index", idx_tmp, "seen too frequently, skipping..." )
                    excess = True
                    break
            if excess:
                continue
            
            # create and save yolt images and labels
            if not (outdir_ims and outdir_labels):
                return
    
            # get yolt labels
            #if verbose:
            #    print ("  Creating yolt labels..."
            yolt_coords = []
            for row in obj_list:
                [index_nest, cat_nest, x0, y0, x1, y1] = row
                yolt_row = [cat_nest] + list(convert((w,h), [x0,x1,y0,y1]))
                # cat_idx = cat_idx_dic[cat_nest]
                # yolt_row = [cat_idx, cat_nest] + list(convert.convert((w,h), [x0,x1,y0,y1]))
                yolt_coords.append(yolt_row)
            if super_verbose:
                print ("   yolt_coords:", yolt_coords )
            
            # # if desired, and randomly selected, flip vertically
            # use_flip = False
            # flip_rand_val = random.uniform(0, 1)
            # if (flip_vert_prob > 0) and (flip_rand_val < flip_vert_prob):
            #     use_flip = True
            #     # flip yolt coords ((cat, x, y, w, h))
            #     yolt_coords_flip = []
            #     for yc in yolt_coords:
            #         yolt_coords_flip.append([yc[0], yc[1], 1.0 - yc[2], yc[3], yc[4]])
            #     yolt_coords = yolt_coords_flip
            #     # flip image
            #     window = np.flipud(window)
            #     flip_suff = '_ud'
            # else:
            #     flip_suff =''
            
            # save image
            skimage.io.imsave(image_outfile, window)
            # cv2.imwrite(image_outfile, window)

            # save labels
            txt_outfile = open(label_outfile, "w")
            for j, yolt_row in enumerate(yolt_coords):
                cls_id = yolt_row[0]
                bb = yolt_row[1:]
                # bb = yolt_row[2:]
                outstring = str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n'
                # print("outstring:", outstring
                txt_outfile.write(outstring)
            txt_outfile.close()             

            # update idx_count_dic
            for idx_tmp in idxs_list:
                idx_count_dic[idx_tmp] += 1
            # update win_iter
            win_iter += 1
                
    # plot to make sure labels and images are created correctly
    if super_verbose:
        print ("\nPlotting yolt training bounding boxes..." )
        print ("  outdir_labels:", outdir_labels )
        print ("  outdir_ims:", outdir_ims )
    if not overwrite:
        plot_training_bboxes(outdir_labels, outdir_ims, 
                 ignore_augment=True,
                 figsize=(10,10), color=(0,0,255), thickness=2, 
                 max_plots=max_plots, sample_label_vis_dir=outdir_yolt_plots,
                 ext=yolt_image_ext, show_plot=False, 
                 specific_labels=[], label_dic=label_str_dict, output_width=500,
                 shuffle=True, verbose=super_verbose) 

    return


###############################################################################
def yolt_from_df_v0(im_path, df_polys, 
                    window_size=512, 
                    jitter_frac=0.1,
                    min_obj_frac=0.7,
                    max_obj_count=1000,
                    geometry_col='geometry', 
                    category_col='da_make_id',
                    image_fname_col='image_fname',
                    outdir_ims=None, 
                    outdir_labels=None,
                    outdir_plots=None, 
                    outdir_yolt_plots=None,
                    aug_count_dict=None,
                    yolt_image_ext='.jpg',
                    max_plots=10,
                    verbose=True, super_verbose=False):
    '''
    Extract yolt cutouts and labels from a singe image.
    df_polys is created from: DA_Dataset/gj_to_px_bboxes.ipynb, which converts geojsons to csvs with columns:
    image_fname, cat_id, loc_id, location, original_make, make, da_make_id, pnp_id, geometry    
    '''
    
    # ensure image exists
    if not os.path.exists(im_path):
        print (" Image file {} DNE...".format(im_path) )
        return
    else:
        im = skimage.io.imread(im_path)
        im_name = os.path.basename(im_path)
        im_root = im_name.split('.')[0]
        image_h, image_w = im.shape[:2]
    # filter dataframe 
    df_im_tmp = df_polys[df_polys[image_fname_col] == im_name]
    
    if verbose:
        print("image", im_name)
        print("  im.shape:", im.shape)
        print("  object count:", df_im_tmp[category_col].value_counts().to_dict())
        # print("  len df_im:", len(df_im_tmp))
    
    # get window cutouts centered at each object
    window_geoms, window_geoms_aug = get_window_geoms(df_im_tmp, 
                                    window_size=window_size, 
                                    image_w=image_w, image_h=image_h,
                                    jitter_frac=jitter_frac, 
                                    geometry_col=geometry_col, 
                                    category_col=category_col,
                                    aug_count_dict=aug_count_dict,
                                    verbose=super_verbose)

    # set up dicts for counting objects
    idx_count_dic = {}
    for idx_tmp in df_im_tmp.index:
        idx_count_dic[idx_tmp] = 0
    idx_count_tot_dic = {}
    for idx_tmp in df_im_tmp.index:
        idx_count_tot_dic[idx_tmp] = 0  
        
    # get objects in each window
    win_iter = 0
    for i, window_geom in enumerate(window_geoms):

        (minx_win, miny_win, maxx_win, maxy_win) = window_geom.bounds
        
        # get window
        window = get_image_window(im, window_geom)
        h, w = window.shape[:2]
        if (h==0) or (w==0):
            continue

        # get objects in window
        obj_list = get_objs_in_window(df_im_tmp, window_geom, 
                                      min_obj_frac=min_obj_frac,
                                      geometry_col=geometry_col, 
                                      category_col=category_col,
                                      use_box_geom=True,
                                      verbose=super_verbose)  
        if super_verbose:
            print ("\nWindow geom:", window_geom )
            print ("  window shape:", window.shape )
            print ("  obj_list:", obj_list )
    
        if len(obj_list) > 0 :
            
            # update idx_count_tot_dic
            idxs_list = [z[0] for z in obj_list]
            for idx_tmp in idxs_list:
                idx_count_tot_dic[idx_tmp] += 1
                
            # Check idx count dic.  If an object has appeared too frequently,
            #   skip the window
            excess = False
            for idx_tmp in idxs_list:
                if idx_count_dic[idx_tmp] >= max_obj_count:
                    print ("Index", idx_tmp, "seen too frequently, skipping..." )
                    excess = True
                    break
            if excess:
                continue
            
            # create and save yolt images and labels
            outroot =  im_root + '__' + 'x0_' + str(int(minx_win)) + '_y0_' \
                                          + str(int(miny_win)) + '_dxdy_' \
                                          + str(int(window_size))
            
            if not (outdir_ims and outdir_labels):
                return
            
            image_outfile = os.path.join(outdir_ims, outroot + yolt_image_ext)
            label_outfile = os.path.join(outdir_labels, outroot + '.txt')
                
            # get yolt labels
            #if verbose:
            #    print ("  Creating yolt labels..."
            yolt_coords = []
            for row in obj_list:
                [index_nest, cat_nest, x0, y0, x1, y1] = row
                yolt_row = [cat_nest] + list(convert((w,h), [x0,x1,y0,y1]))
                # cat_idx = cat_idx_dic[cat_nest]
                # yolt_row = [cat_idx, cat_nest] + list(convert.utils.convert((w,h), [x0,x1,y0,y1]))
                yolt_coords.append(yolt_row)
            if super_verbose:
                print ("   yolt_coords:", yolt_coords )

            # try flipping, rotating if desired, see augment func above...
            # ...
            
            # save image
            # if verbose:
            #     print ("  Saving window to file..." )
            #     print ("    window.dtype:", window.dtype )
            #     print ("    window.shape:", window.shape )
            skimage.io.imsave(image_outfile, window)
            # cv2.imwrite(image_outfile, window)

            # save labels
            txt_outfile = open(label_outfile, "w")
            for j, yolt_row in enumerate(yolt_coords):
                cls_id = yolt_row[0]
                bb = yolt_row[1:]
                # bb = yolt_row[2:]
                outstring = str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n'
                # print("outstring:", outstring
                txt_outfile.write(outstring)
            txt_outfile.close()             

            # # make plots
            # plot_outfile = os.path.join(outdir_plots,  outroot + yolt_image_ext)
            # if i <= args.max_plots:
            #     if verbose:
            #         print ("obj_list:",obj_list )
            #         print ("plot outfile:", plot_outfile )
            #     #im_copy = im.copy()
            #     #plot_obj_list(im_copy, window_geom, obj_list, color_dic, 
            #     plot_obj_list(window.copy(), obj_list, color_dic, 
            #                   thickness=plot_thickness, show_plot=False, 
            #                   outfile=plot_outfile) 
                
            # update idx_count_dic
            for idx_tmp in idxs_list:
                idx_count_dic[idx_tmp] += 1
            # update win_iter
            win_iter += 1


    # # augment, if desired
    # if args.augment.upper() == 'TRUE':
    #     yolt_data_prep_funcs.augment_training_data(outdir_labels, outdir_ims, 
    #                                                hsv_range=[0.5,1.5],
    #                                                ext=ext,
    #                                                skip_hsv_transform=False)
                
    # make sure labels and images are created correctly
    print ("\nPlotting yolt training bounding boxes..." )
    print ("  outdir_labels:", outdir_labels )
    print ("  outdir_ims:", outdir_ims )
    plot_training_bboxes(outdir_labels, outdir_ims, 
                 ignore_augment=True,
                 figsize=(10,10), color=(0,0,255), thickness=2, 
                 max_plots=max_plots, sample_label_vis_dir=outdir_yolt_plots,
                 ext=yolt_image_ext, show_plot=False, 
                 specific_labels=[], label_dic=[], output_width=500,
                 shuffle=True, verbose=super_verbose) 
    
    return


###############################################################################
def get_labels(csv_path, xmin, ymin, width, height, label_col, min_overlap=0, classes=True):
        """Credit: Nick Weir
        The function below will retrieve labels from AOI csvs given:
        `csv_path`, `xmin`, `ymin`, `width`, `height`, and `min_overlap`. 
            `label_col` specifies which column contains integral label IDs.

        Read in labels from a pixel coordinate csv input.
        
        Notes
        -----
        WARNING: This function will not raise any kind of error if the x, y, width,
        or height values selected don't fall within the bounds of the image. Make
        sure that the values you select fall within the pixel extent of the image
        before using.
        
        Arguments
        ---------
        csv_path : str
            The path to the CSV file containing geometries for an AOI of interest.
        xmin : int
            The starting X pixel index for the tile to be extracted. 
        ymin : int
            The starting Y pixel index for the tile to be extracted.
        width : int
            The width of the tile to be extracted, in pixel units.
        height : int
            The height of the tile to be extracted, in pixel units.
        label_col : str
            The name of the column in the CSV containing integral label IDs.
        min_overlap : float, optional
            The fraction of a geometry area that must overlap with the extracted
            tile for the label to be included in the output. Defaults to 0 (any
            portion of an overlapping geometry, however small, will be labeled in
            the output).
        
        Returns
        -------
        geom_bounds : :class:`numpy.ndarray`
            A NumPy array of shape ``(n_labels, 4)``. Each row corresponds to a single
            label's ``[xmin, ymin, xmax, ymax]`` coordinates. The units are the fraction
            of the image's extent in the given direction (width for ``xmin`` and ``xmax``,
            height for ``ymin`` and ``ymax``.)
        """
        # needs to be read as a gdf for spatial operations
        gdf = gpd.GeoDataFrame(pd.read_csv(csv_path))
        gdf.geometry = gdf.geometry.apply(loads)
        tile_bbox = box(xmin, ymin, xmin+width, ymin+height)

        to_keep = gdf.geometry.apply(lambda x: tile_bbox.intersects(x))
        kept_inds = gdf.index[to_keep].values
        tile_gdf = gdf.loc[to_keep, :]
        frac_overlap = tile_gdf.geometry.apply(lambda x: tile_bbox.intersection(x).area/x.area)
        if len(frac_overlap) == 0:
            return [], [], []
        tile_gdf = tile_gdf.loc[frac_overlap > min_overlap, :]
        if len(tile_gdf) == 0:
            return [], [], []
        tile_gdf.geometry = tile_gdf.geometry.apply(lambda x: tile_bbox.intersection(x))
        tile_gdf.geometry = tile_gdf.geometry.apply(
            translate, xoff=-xmin, yoff=-ymin)

        geom_bounds = tile_gdf.bounds
        geom_bounds['minx'] = geom_bounds['minx'].apply(
            lambda x: x/width)
        geom_bounds['maxx'] = geom_bounds['maxx'].apply(
            lambda x: x/width)
        geom_bounds['miny'] = geom_bounds['miny'].apply(
            lambda y: y/height)
        geom_bounds['maxy'] = geom_bounds['maxy'].apply(
            lambda y: y/height)
        geom_bounds['width'] = geom_bounds['maxx'] - geom_bounds['minx']
        geom_bounds['midx'] = geom_bounds['minx'] + geom_bounds['width']/2
        geom_bounds['height'] = geom_bounds['maxy'] - geom_bounds['miny']
        geom_bounds['midy'] = geom_bounds['miny'] + geom_bounds['height']/2 
        
        geom_bounds = np.around(geom_bounds[['midx', 'midy', 'width', 'height']].to_numpy(), 6).tolist()
        if classes:
            labels = tile_gdf[label_col].tolist()
        else:
            labels = len(tile_gdf)

        return geom_bounds, labels, kept_inds


###############################################################################
def map_wrapper(x):
    '''For multi-threading'''
    return x[0](*(x[1:]))


###############################################################################
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """

    weighted_stats = DescrStatsW(values, weights=weights, ddof=0)

    # weighted mean of data (equivalent to np.average(array, weights=weights))
    mean = weighted_stats.mean
    # standard deviation with default degrees of freedom correction
    std = weighted_stats.std
    # variance with default degrees of freedom correction
    var = weighted_stats.var

    return (mean, std, var)

def slice_im_plus_boxes(image_path, out_name, out_dir_images, 
             boxes=[], yolo_classes=[], out_dir_labels=None, 
             mask_path=None, out_dir_masks=None,
             sliceHeight=416, sliceWidth=416,
             overlap=0.1, slice_sep='|', pad=0,
             skip_highly_overlapped_tiles=False,
             overwrite=False,
             out_ext='.png', verbose=False):

    """
    Slice a large image into smaller windows, and also bin boxes
    Adapted from:
         https://github.com/avanetten/simrdwn/blob/master/simrdwn/core/slice_im.py

    Arguments
    ---------
    image_path : str
        Location of image to slice
    out_name : str
        Root name of output files (coordinates will be appended to this)
    out_dir_images : str
        Output directory for images
	boxes : arr
		List of bounding boxes in image, in pixel coords
        [ [xb0, yb0, xb1, yb1], ...]
        Defaults to []
    yolo_classes : list
        list of class of objects for each box [0, 1, 0, ...]
        Defaults to []
    out_dir_labels : str
        Output directory for labels
        Defaults to None
    sliceHeight : int
        Height of each slice.  Defaults to ``416``.
    sliceWidth : int
        Width of each slice.  Defaults to ``416``.
    overlap : float
        Fractional overlap of each window (e.g. an overlap of 0.2 for a window
        of size 256 yields an overlap of 51 pixels).
        Default to ``0.1``.
    slice_sep : str
        Character used to separate outname from coordinates in the saved
        windows.  Defaults to ``|``
    out_ext : str
        Extension of saved images.  Defaults to ``.png``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``

    Returns
    -------
    None
    """

    if len(out_ext) == 0:
        im_ext = '.' + image_path.split('.')[-1]
    else:
        im_ext = out_ext

    t0 = time.time()
    image = skimage.io.imread(image_path)  #, as_grey=False).astype(np.uint8)  # [::-1]
    print("image.shape:", image.shape)
    if mask_path:
        mask = skimage.io.imread(mask_path)
    win_h, win_w = image.shape[:2]
    win_size = sliceHeight*sliceWidth
    dx = int((1. - overlap) * sliceWidth)
    dy = int((1. - overlap) * sliceHeight)
    
    n_ims = 0
    for y0 in range(0, image.shape[0], dy):
        for x0 in range(0, image.shape[1], dx):
            out_boxes_yolo = []
            out_classes_yolo = []
            n_ims += 1

            if (n_ims % 100) == 0:
                print(n_ims)

            # make sure we don't have a tiny image on the edge
            if y0+sliceHeight > image.shape[0]:
                # skip if too much overlap (> 0.6)
                if skip_highly_overlapped_tiles:
                    if (y0+sliceHeight - image.shape[0]) > (0.6*sliceHeight):
                        continue
                    else:
                        y = image.shape[0] - sliceHeight
                else:
                    y = image.shape[0] - sliceHeight
            else:
                y = y0
            if x0+sliceWidth > image.shape[1]:
                # skip if too much overlap (> 0.6)
                if skip_highly_overlapped_tiles:
                    if (x0+sliceWidth - image.shape[1]) > (0.6*sliceWidth):
                        continue
                    else:
                        x = image.shape[1] - sliceWidth
                else:
                    x = image.shape[1] - sliceWidth
            else:
                x = x0

            xmin, xmax, ymin, ymax = x, x+sliceWidth, y, y+sliceHeight

            # find boxes that lie entirely within the window
            if len(boxes) > 0:
                out_path_label = os.path.join(
                    out_dir_labels,
                    out_name + slice_sep + str(y) + '_' + str(x) + '_'
                    + str(sliceHeight) + '_' + str(sliceWidth)
                    + '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h)
                    + '.txt')
                for j,b in enumerate(boxes):
                    yolo_class = yolo_classes[j]
                    xb0, yb0, xb1, yb1 = b
                    if (xb0 >= xmin) and (yb0 >= ymin) \
                        and (xb1 <= xmax) and (yb1 <= ymax):
                        # get box coordinates within window
                        out_box_tmp = [xb0 - xmin, xb1 - xmin,
                                       yb0 - ymin, yb1 - ymin]
                        print("  out_box_tmp:", out_box_tmp)
                        # out_boxes.append(out_box_tmp)
                        # convert to yolo coords (x,y,w,h)
                        yolo_coords = convert((sliceWidth, sliceHeight),
                                               out_box_tmp)
                        print("    yolo_coords:", yolo_coords)
                        out_boxes_yolo.append(yolo_coords)
                        out_classes_yolo.append(yolo_class)
            
                # skip if no labels?
                if len(out_boxes_yolo) == 0:
                    continue

                # save yolo labels
                txt_outfile = open(out_path_label, "w")     
                for yolo_class, yolo_coord in zip(out_classes_yolo, out_boxes_yolo):                          
                    outstring = str(yolo_class) + " " + " ".join([str(a) for a in yolo_coord]) + '\n'
                    if verbose: 
                         print("  outstring:", outstring.strip())
                    txt_outfile.write(outstring)
                txt_outfile.close()                

            # save mask, if desired
            if mask_path:
                mask_c = mask[y:y + sliceHeight, x:x + sliceWidth]
                outpath_mask = os.path.join(
                    out_dir_masks,
                    out_name + slice_sep + str(y) + '_' + str(x) + '_'
                    + str(sliceHeight) + '_' + str(sliceWidth)
                    + '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h)
                    + im_ext)
                skimage.io.imsave(outpath_mask, mask_c, check_contrast=False)

            # extract image
            window_c = image[y:y + sliceHeight, x:x + sliceWidth]
            outpath = os.path.join(
                out_dir_images,
                out_name + slice_sep + str(y) + '_' + str(x) + '_'
                + str(sliceHeight) + '_' + str(sliceWidth)
                + '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h)
                + im_ext)
            if not os.path.exists(outpath):
                skimage.io.imsave(outpath, window_c, check_contrast=False)
            elif overwrite:
                skimage.io.imsave(outpath, window_c, check_contrast=False)
            else:
                print("outpath {} exists, skipping".format(outpath))
                                                                                                 
    print("Num slices:", n_ims,
          "sliceHeight", sliceHeight, "sliceWidth", sliceWidth)
    print("Time to slice", image_path, time.time()-t0, "seconds")
    
    return
###############################################################################
def twinx_function(x, raw=False):
    V = 3./x
    if raw:
        return V
    else:
        return ["%.1f" % z for z in V]
    # return [z for z in V]


###############################################################################
def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x,
                        [x < x0],
                        [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])


###############################################################################
def _file_len(fname):
    '''Return length of file'''
    try:
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    except:
        return 0


###############################################################################
def _run_cmd(cmd):
    '''Write to stdout, etc,(incompatible with nohup)'''
    p = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
    while True:
        line = p.stdout.readline()
        if not line:
            break
        print(line.replace('\n', ''))
    return



# https://github.com/CosmiQ/simrdwn/blob/master/simrdwn/data_prep/yolt_data_prep_funcs.py
###############################################################################
def convert(size, box):
    '''Input = image size: (w,h), box: [x0, x1, y0, y1]
    Return yolo coords: normalized (x, y, w, h)'''
    dw = 1./size[0]
    dh = 1./size[1]
    xmid = (box[0] + box[1])/2.0
    ymid = (box[2] + box[3])/2.0
    w0 = box[1] - box[0]
    h0 = box[3] - box[2]
    x = xmid*dw
    y = ymid*dh
    w = w0*dw
    h = h0*dh
    return (x,y,w,h)
 
 
# https://github.com/CosmiQ/simrdwn/blob/master/simrdwn/data_prep/yolt_data_prep_funcs.py
###############################################################################
def convert_reverse(size, box):
    '''Back out pixel coords from yolo format
    input = image_size (w,h), 
        box = [x,y,w,h]'''
    x, y, w, h = box
    dw = 1./size[0]
    dh = 1./size[1]

    w0 = w/dw
    h0 = h/dh
    xmid = x/dw
    ymid = y/dh

    x0, x1 = xmid - w0/2., xmid + w0/2.
    y0, y1 = ymid - h0/2., ymid + h0/2.

    return [x0, x1, y0, y1]

###############################################################################

    
    
###############################################################################
def slice_im_plus_df(image_path, out_name, out_dir_images, 
             label_df=None, 
             min_obj_frac=0.7, 
             geometry_col='geometry_poly_pixel', 
             category_col='Category',
             out_dir_geojson=None,
             sliceHeight=416, sliceWidth=416,
             overlap=0.1, slice_sep='|', pad=0,
             skip_highly_overlapped_tiles=False,
             overwrite=False,
             keep_empty_geojsons=False,
             out_ext='.png', verbose=False):

    """
    Slice a large image into smaller windows, and also return boxes labels 
             withing the window (label_df is a geojson of labels)
    Adapted from:
         https://github.com/avanetten/simrdwn/blob/master/simrdwn/core/slice_im.py

    Arguments
    ---------
    image_path : str
        Location of image to slice
    out_name : str
        Root name of output files (coordinates will be appended to this)
    out_dir_images : str
        Output directory for images
	label_df : dataframe
		dataframe of labels
    sliceHeight : int
        Height of each slice.  Defaults to ``416``.
    sliceWidth : int
        Width of each slice.  Defaults to ``416``.
    overlap : float
        Fractional overlap of each window (e.g. an overlap of 0.2 for a window
        of size 256 yields an overlap of 51 pixels).
        Default to ``0.1``.
    slice_sep : str
        Character used to separate outname from coordinates in the saved
        windows.  Defaults to ``|``
    out_ext : str
        Extension of saved images.  Defaults to ``.png``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``

    Returns
    -------
    None
    """

    if len(out_ext) == 0:
        im_ext = '.' + image_path.split('.')[-1]
    else:
        im_ext = out_ext

    t0 = time.time()
    image = skimage.io.imread(image_path)  #, as_grey=False).astype(np.uint8)  # [::-1]
    if verbose:
        print("image.shape:", image.shape)
    win_h, win_w = image.shape[:2]
    win_size = sliceHeight*sliceWidth
    dx = int((1. - overlap) * sliceWidth)
    dy = int((1. - overlap) * sliceHeight)
    
    n_ims = 0
    for y0 in range(0, image.shape[0], dy):
        for x0 in range(0, image.shape[1], dx):
            out_boxes_yolo = []
            out_classes_yolo = []
            n_ims += 1

            if (n_ims % 100) == 0:
                print(n_ims)

            # make sure we don't have a tiny image on the edge
            if y0+sliceHeight > image.shape[0]:
                # skip if too much overlap (> 0.6)
                if skip_highly_overlapped_tiles:
                    if (y0+sliceHeight - image.shape[0]) > (0.6*sliceHeight):
                        continue
                    else:
                        y = image.shape[0] - sliceHeight
                else:
                    y = image.shape[0] - sliceHeight
            else:
                y = y0
            if x0+sliceWidth > image.shape[1]:
                # skip if too much overlap (> 0.6)
                if skip_highly_overlapped_tiles:
                    if (x0+sliceWidth - image.shape[1]) > (0.6*sliceWidth):
                        continue
                    else:
                        x = image.shape[1] - sliceWidth
                else:
                    x = image.shape[1] - sliceWidth
            else:
                x = x0

            xmin, xmax, ymin, ymax = x, x+sliceWidth, y, y+sliceHeight
            
            # extract labels from window
            if label_df is not None:
                # get geom of window (see prep_train.get_window_geoms())
                win_p1 = shapely.geometry.Point(xmin, ymin)
                win_p2 = shapely.geometry.Point(xmax, ymin)
                win_p3 = shapely.geometry.Point(xmax, ymax)
                win_p4 = shapely.geometry.Point(xmin, ymax)
                pointList = [win_p1, win_p2, win_p3, win_p4, win_p1]
                geom_window = shapely.geometry.Polygon([[p.x, p.y] for p in pointList])
            
                # get objects within the window
                # obj_list has form: [[index_nest, cat_nest, x0_obj, y0_obj, x1_obj, y1_obj], ...]
                obj_list = get_objs_in_window(label_df, geom_window, 
                        min_obj_frac=min_obj_frac, 
                        geometry_col=geometry_col, category_col=category_col,
                        use_box_geom=True, verbose=False)

                # create output gdf, geojson
                outpath_geojson = os.path.join(
                        out_dir_geojson,
                        out_name + slice_sep + str(y) + '_' + str(x) + '_'
                        + str(sliceHeight) + '_' + str(sliceWidth)
                        + '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h)
                        + '.geojson')
                if len(obj_list) == 0:
                    if keep_empty_geojsons:
                        if out_dir_geojson:
                            # print("Empty dataframe, writing empty gdf", output_path)
                            open(outpath_geojson, 'a').close()
                else:
                    index_l, cat_l, geom_l = [], [], []
                    for row in obj_list:
                        index_nest, cat_nest, xmin_tmp, ymin_tmp, xmax_tmp, ymax_tmp = row
                        tmp_p1 = shapely.geometry.Point(xmin_tmp, ymin_tmp)
                        tmp_p2 = shapely.geometry.Point(xmax_tmp, ymin_tmp)
                        tmp_p3 = shapely.geometry.Point(xmax_tmp, ymax_tmp)
                        tmp_p4 = shapely.geometry.Point(xmin_tmp, ymax_tmp)
                        pointList = [tmp_p1, tmp_p2, tmp_p3, tmp_p4, tmp_p1]
                        geom_tmp = shapely.geometry.Polygon([[p.x, p.y] for p in pointList])
                        index_l.append(index_nest)
                        cat_l.append(cat_nest)
                        geom_l.append(geom_tmp)
                    # construct dataframe
                    dict_tmp = {'index_nest': index_l, category_col: cat_l, geometry_col: geom_l}
                    gdf_tmp = gpd.GeoDataFrame(dict_tmp)
                    # print("gdf_tmp:", gdf_tmp)
                    # save to geojson, if desired
                    if out_dir_geojson:
                        gdf_tmp.to_file(outpath_geojson, driver='GeoJSON')

            # extract image
            window_c = image[y:y + sliceHeight, x:x + sliceWidth]
            outpath = os.path.join(
                out_dir_images,
                out_name + slice_sep + str(y) + '_' + str(x) + '_'
                + str(sliceHeight) + '_' + str(sliceWidth)
                + '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h)
                + im_ext)
            if not os.path.exists(outpath):
                skimage.io.imsave(outpath, window_c, check_contrast=False)
            elif overwrite:
                skimage.io.imsave(outpath, window_c, check_contrast=False)
            else:
                if verbose:
                    print("outpath {} exists, skipping".format(outpath))
                                                                                                 
    print("Num slices:", n_ims,
          "sliceHeight", sliceHeight, "sliceWidth", sliceWidth)
    print("Time to slice", image_path, time.time()-t0, "seconds")
    return

