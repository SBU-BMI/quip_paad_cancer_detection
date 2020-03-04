import numpy as np
import openslide
import sys
import os
from PIL import Image
import datetime
import time
#from stain_norm_python.color_normalize_single_folder import color_normalize_single_folder
import sys
import cv2
import multiprocessing as mp
import random

slide_name = sys.argv[2] + '/' + sys.argv[1];
output_folder = sys.argv[3] + '/' + sys.argv[1];
patch_size_5X = 1050;
level = 1

#start = time.time()
#time.sleep(random.randint(100, 1000)/100.0)  # wait for 1 --> 10s to avoid concurrency

fdone = '{}/extraction_done.txt'.format(output_folder);
if os.path.isfile(fdone):
    print('fdone {} exist, skipping'.format(fdone));
    exit(0);

print('extracting {}'.format(output_folder));

if not os.path.exists(output_folder):
    os.mkdir(output_folder);

try:
    oslide = openslide.OpenSlide(slide_name);
    if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
        mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
    elif "XResolution" in oslide.properties:
        mag = 10.0 / float(oslide.properties["XResolution"]);
    elif 'tiff.XResolution' in oslide.properties:   # for Multiplex IHC WSIs, .tiff images
        Xres = float(oslide.properties["tiff.XResolution"])
        if Xres < 10:
            mag = 10.0 / Xres;
        else:
            mag = 10.0 / (10000/Xres)       # SEER PRAD
    else:
        mag = 10.0 / float(0.254);
        print('[WARNING] mpp value not found. Assuming it is 40X with mpp=0.254!', slide_name);
        
    print('mag: ', mag)
    pw = int(patch_size_5X * mag / 6.6666667);
    width = oslide.dimensions[0];
    height = oslide.dimensions[1];
    scale_down = oslide.level_downsamples[level]
except:
    print('{}: exception caught'.format(slide_name));
    exit(1);

print('slide/width/height/scale_down/pw: ', slide_name, width, height, scale_down, pw)
sys.stdout.flush()

corrs = []

for x in range(1, width, pw):
    for y in range(1, height, pw):
        if x + pw > width:
            pw_x = width - x;
        else:
            pw_x = pw;

        if y + pw > height:
            pw_y = height - y;
        else:
            pw_y = pw;

        if int(patch_size_5X * pw_x / pw) > 50 and int(patch_size_5X * pw_y / pw) > 50:
            corrs.append((x, y, pw_x, pw_y))

def extract_patch(corr):
    x, y, pw_x, pw_y = corr

    patch = oslide.read_region((x, y), level, (int(pw_x/scale_down), int(pw_y/scale_down)));
    patch = patch.resize((int(patch_size_5X * pw_x / pw), int(patch_size_5X * pw_y / pw)), Image.ANTIALIAS);
    fname = '{}/{}_{}_{}_{}.png'.format(output_folder, x, y, pw, patch_size_5X);
    patch.save(fname);


print(slide_name, len(corrs))
pool = mp.Pool(processes=4)
pool.map(extract_patch, corrs)

print('Elapsed time: ', (time.time() - start)/60.0)
#open(fdone, 'w').close();

