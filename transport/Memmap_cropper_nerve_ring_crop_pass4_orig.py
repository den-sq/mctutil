import tifffile
from pathlib import Path
import sys
import os
import glob
import datetime
import numpy as np
from multiprocessing import Pool
import pathlib
import cv2
from scipy import ndimage
from datetime import datetime
from scipy import interpolate


startTime = datetime.now()
proj_dir = os.path.dirname(sys.argv[0])
bits = '16'
circle_mask = True
skew_angle -= 54
output_dir = os.path.join(proj_dir, 'Octo_9_pass4_nerve_ring_crop')
Path(output_dir).mkdir(parents=True, exist_ok=True)


def get_image_paths(folder):
    return (sorted(glob.glob(folder + '\\*.tif*')))


def bit_level(image_path):
    img=tifffile.imread(image_path)
    img[img<0]=0
    
    if bits=='16':
        img = (65536*(img - norm_stat_min)/norm_stat_ptp).astype('uint16')
    if bits=='8':
        img = (255*(img - norm_stat_min)/norm_stat_ptp).astype('uint8')
    tifffile.imwrite(os.path.join(output_dir,os.path.basename(image_path)),img,compression=1)
    print(f'processed {image_path}')
    
def bit_level_crop_rotate(image_path):
    img=tifffile.memmap(image_path)#[4200:6500,6000:8400]
 
    tifffile.imwrite(os.path.join(output_dir,os.path.basename(image_path)),img[4200:7300,4400:7920])
    print(f'processed {image_path}')


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(h/2), int(w/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], h-center[0], w-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((Y - center[0])**2 + (X-center[1])**2)

    mask = dist_from_center <= radius
    return mask
    
    
    
def parallel_circle_mask_apply(image_path):
    img=tifffile.imread(image_path)
    ydim,xdim=img.shape
    circle_center=(int(ydim/2),int(xdim/2))
    mask=create_circular_mask(ydim,xdim,center=circle_center,radius=2800)
    img[~mask]=0
    tifffile.imwrite(os.path.join(output_dir,os.path.basename(image_path)),img)



# input_dir_=os.path.join('reconstructed','AAA688_14keV','32bit_AAA688_14keV_Ping_He')
# input_dir_=os.path.join('32bit_reconstructed_bin4_scan_1671238250_9001projs_60000ussection1_scan_tiff_180_pass3_BAC_2.0_2.0')
input_dir=sys.argv[1]
file_names=get_image_paths(input_dir)
file_names=file_names[2430:2775]
if __name__ == '__main__':
    startTime_full = datetime.now()


    
  
    pool = Pool(16)
    pool.map(bit_level_crop_rotate, file_names)
    pool.close()
    pool.join()

    print('\nfull time ' + str(datetime.now()-startTime_full))