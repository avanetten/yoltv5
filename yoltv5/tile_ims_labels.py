#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 2020
@author: avanetten
"""


import os
import cv2
import time
import shapely
import numpy as np
import skimage.io

import prep_train

###############################################################################
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
                        yolo_coords = prep_train.convert((sliceWidth, sliceHeight),
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
                obj_list = prep_train.get_objs_in_window(label_df, geom_window, 
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