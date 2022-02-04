#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 12:22:36 2021

@author: avanetten

Script to execute yoltv5 testing
"""

import pandas as pd
import skimage.io
import argparse
import shutil
import yaml
import time
import sys
import os

######################################
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
######################################
# 0. Load config and set variables
######################################

parser = argparse.ArgumentParser()
parser.add_argument('config_path')
args = parser.parse_args()

with open(args.config_path, 'r') as f:
    config_dict = yaml.safe_load(f)
    f.close()
config = dotdict(config_dict)
print("test.py: config:")
print(config)


######################################
# 1. Import yoltv5 scripts
######################################

yolt_src_path = os.path.join(config.yoltv5_path, 'yoltv5')
print("yoltv5_execute_test.py: yolt_src_path:", yolt_src_path)
sys.path.append(yolt_src_path)
import prep_train
import tile_ims_labels
import post_process
import eval
import eval_errors

# check if output already exists
results_dir = os.path.join(yolt_src_path, 'yolov5/runs/detect', config.outname_infer)
if os.path.exists(results_dir):
    raise ValueError('Breaking, since output directory already exists {}'.format(results_dir))

######################################
# 2. Prepare data
######################################
t0 = time.time()

###################
# object names
###################
# create name file
cat_int_to_name_dict = {}
namefile = os.path.join(config.yoltv5_path, 'data', config.name_file_name)
for i, n in enumerate(config.object_names):
    cat_int_to_name_dict[i] = n
    if i == 0:
        os.system( 'echo {} > {}'.format(n, namefile))
    else:
        os.system( 'echo {} >> {}'.format(n, namefile))
# view
print("\nobject names ({})".format(namefile))
with open(namefile,'r') as f:
    all_lines = f.readlines()
    for l in all_lines:
        print(l)
print("cat_int_to_name_dict:", cat_int_to_name_dict)

###################
# slice test images
###################
if config.sliceWidth > 0:
    # # make list of test files
    print("\nslicing im_dir:", config.test_im_dir)
    im_list = [z for z in os.listdir(config.test_im_dir) if z.endswith(config.im_ext)]
    if not os.path.exists(config.outdir_slice_ims):
        os.makedirs(config.outdir_slice_ims) #, exist_ok=True)
        os.makedirs(config.outdir_slice_txt) #, exist_ok=True)
        print("outdir_slice_ims:", config.outdir_slice_ims)
        # slice images
        for i,im_name in enumerate(im_list):
            im_path = os.path.join(config.test_im_dir, im_name)
            im_tmp = skimage.io.imread(im_path)
            h, w = im_tmp.shape[:2]
            print(i, "/", len(im_list), im_name, "h, w =", h, w)

            # tile data
            out_name = im_name.split('.')[0]
            tile_ims_labels.slice_im_plus_boxes(
                im_path, out_name, config.outdir_slice_ims,
                sliceHeight=config.sliceHeight, sliceWidth=config.sliceWidth,
                overlap=config.slice_overlap, slice_sep=config.slice_sep,
                skip_highly_overlapped_tiles=config.skip_highly_overlapped_tiles,
                overwrite=config.slice_overwrite,
                out_ext=config.out_ext, verbose=config.slice_verbose)
        im_list_test = []
        for f in sorted([z for z in os.listdir(config.outdir_slice_ims) if z.endswith(config.out_ext)]):
            im_list_test.append(os.path.join(config.outdir_slice_ims, f))
        df_tmp = pd.DataFrame({'image': im_list_test})
        df_tmp.to_csv(config.outpath_test_txt, header=False, index=False)
    else:
        print("Images already sliced to:", config.outdir_slice_ims)
        df_tmp = pd.read_csv(config.outpath_test_txt, names=['path'])
        im_list_test = list(df_tmp['path'].values)
else:
    # forego slicing
    im_list_test = []
    for f in sorted([z for z in os.listdir(config.test_im_dir) if z.endswith(config.im_ext)]):
        im_list_test.append(os.path.join(config.outdir_ims, f))
    df_tmp = pd.DataFrame({'image': im_list_test})
    df_tmp.to_csv(config.outpath_test_txt, header=False, index=False)
# print some values
print("N test images:", len(im_list))
print("N test slices:", len(df_tmp))
# view
print("head of test files ({})".format(config.outpath_test_txt))
with open(config.outpath_test_txt,'r') as f:
    all_lines = f.readlines()
    for i,l in enumerate(all_lines):
        if i < 5:
            print(l)
        else:
            break
            

######################################
# 3. Execute GPU inference
# time python yolov5/detect.py --weights yolov5/runs/train/exp/weights/last.pt --img 640 --conf 0.25 \
#     --source test_images/images_slice_544 \
#     --nosave --save-txt --save-conf \
#     --name yoltv5_test_v0
# # Results saved to yolov5/runs/detect/yoltv5_test_v0
######################################

script_path = os.path.join(config.yoltv5_path, 'yoltv5/yolov5/detect.py')
yolt_cmd = 'python {} --weights {} --source {} --img {} --conf {} ' \
            '--name {} --nosave --save-txt --save-conf'.format(\
            script_path, config.weights_file, config.outdir_slice_ims,
            config.train_im_size, min(config.detection_threshes), 
            config.outname_infer)
print("\nyolt_cmd:", yolt_cmd)
os.system(yolt_cmd)


######################################
# 4. Post process (CPU)
######################################
pred_dir = os.path.join(results_dir, 'labels')
# pred_dir = os.path.join(config.yoltv5_path, 'yoltv5', 'yolov5/runs/detect', config.outname_infer)
out_dir_root = os.path.join(config.yoltv5_path, 'results', config.outname_infer)
os.makedirs(out_dir_root, exist_ok=True)
print("post-proccessing:", config.outname_infer)
for detection_thresh in config.detection_threshes:

    out_csv = 'preds_refine_' + str(detection_thresh).replace('.', 'p') + '.csv'
    out_geojson_geo_dir = 'geojsons_geo_' + str(detection_thresh).replace('.', 'p')
    out_geojson_pix_dir = 'geojsons_pix_' + str(detection_thresh).replace('.', 'p')
    plot_dir = 'pred_plots_' + str(detection_thresh).replace('.', 'p')
    if config.extract_chips:
        out_dir_chips = 'detection_chips_' + str(detection_thresh).replace('.', 'p')
    else:
        out_dir_chips = ''

    # post_process
    post_process.execute(
        pred_dir=pred_dir,
        truth_file=config.truth_file,
        raw_im_dir=config.test_im_dir,
        out_dir_root=out_dir_root,
        out_csv=out_csv,
        cat_int_to_name_dict=cat_int_to_name_dict,
        ignore_names=config.ignore_names,
        out_geojson_geo_dir=out_geojson_geo_dir,
        out_geojson_pix_dir=out_geojson_pix_dir,
        plot_dir=plot_dir,
        im_ext=config.im_ext,
        out_dir_chips=out_dir_chips,
        chip_ext=config.chip_ext,
        chip_rescale_frac=config.chip_rescale_frac,
        allow_nested_detections=config.allow_nested_detections,
        max_edge_aspect_ratio=config.max_edge_aspect_ratio,
        nms_overlap_thresh=config.nms_overlap_thresh,
        slice_size=config.sliceWidth,
        sep=config.slice_sep,
        n_plots=config.n_plots,
        edge_buffer_test=config.edge_buffer_test,
        max_bbox_size_pix=config.max_bbox_size,
        detection_thresh=detection_thresh)


tf = time.time()
print("\nResults saved to: {}".format(out_dir_root))
print("\nTotal time to run inference and make plots:", tf - t0, "seconds")