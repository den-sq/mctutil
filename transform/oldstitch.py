import sys
import os
import glob

os.environ['NUMEXPR_MAX_THREADS'] = '272'

import numpy as np
import tifffile
from pathlib import Path
from multiprocessing import Pool
from datetime import datetime
import cv2


def get_image_paths(folder):
	return sorted(glob.glob(os.path.join(folder, ' * .tif * ')))


def bit_level_crop_stitch(inlist):
	list_index, filenames_chunk, filenames_chunk_idx, x, y, w, h, bits, aphla_mask_path, norm_stat_min, norm_stat_ptp, \
		out_dir, image_section_list_chunk = inlist
	print(list_index)

	paths = filenames_chunk[list_index]
	image_section_list = image_section_list_chunk[list_index]

	# circle_center = (int(ydim / 2), int(xdim / 2))
	# mask = create_circular_mask(ydim,xdim,center = circle_center,radius = Circle_mask_radius)

	print(f'thread {list_index} starting to stitch {len(paths)} images')
	aphla_mask = tifffile.memmap(aphla_mask_path)
	for idx, image_path in enumerate(paths):
		incoming_proj_map = tifffile.memmap(image_path)
		if bits == '8':
			incoming_proj_map = (65536 * (incoming_proj_map - norm_stat_min) / norm_stat_ptp)
			incoming_proj_map[incoming_proj_map < 0] = 0
			incoming_proj_map[incoming_proj_map > 65535] = 65535
			incoming_proj_map = incoming_proj_map.astype('uint16')
			incoming_proj_map = np.multiply(incoming_proj_map, aphla_mask).astype('uint16')
		else:
			incoming_proj_map = (255 * (incoming_proj_map - norm_stat_min) / norm_stat_ptp)
			incoming_proj_map[incoming_proj_map < 0] = 0
			incoming_proj_map[incoming_proj_map > 255] = 255
			incoming_proj_map = incoming_proj_map.astype('uint8')
			incoming_proj_map = np.multiply(incoming_proj_map, aphla_mask).astype('uint8')

		tifffile.imwrite(os.path.join(out_dir,
									f'{str(list_index).zfill(4)}_{image_section_list[idx]}_{os.path.basename(image_path)}'),
									incoming_proj_map[y - crop_buffer_px: y + h + crop_buffer_px,
														x - crop_buffer_px: x + w + crop_buffer_px])
		out_path = os.path.join(out_dir, f'{list_index}_{os.path.basename(image_path)}')
		print(f'done stitching image {image_path} to {out_path} total: {idx} / {len(image_path)}')


def reject_outliers(data, m=3):
	data1 = abs(data - np.mean(data))
	stat = m * np.std(data)
	data[data1 > stat] = 0
	return data


def bit_level_crop(inlist):
	image_path, x, y, w, h, bits, alpha_blurr = inlist
	img = tifffile.imread(image_path)
	img = img[final_recon_crop]
	alpha_blurr_32bit = np.divide(alpha_blurr, 255, dtype=np.float32)
	alpha_blurr_32bit = alpha_blurr_32bit[y - crop_buffer_px:
											y + h + crop_buffer_px, x - crop_buffer_px: x + w + crop_buffer_px]

	img = img[y - crop_buffer_px: y + h + crop_buffer_px, x - crop_buffer_px: x + w + crop_buffer_px]
	img[img < 0] = 0
	scale_Function = False
	if scale_Function:
		img = win_scale(img, .0008, .0017, f'uint{bits}', [0, 10000])
	else:
		if bits == '16':
			img = (65536 * (img - norm_stat_min) / norm_stat_ptp * ptp_scale_factor)
			img[img < 0] = 0
			img[img > 65535] = 65535
			img = img.astype('uint16')
			# img = (65536 * (img - .000305) / .00145)
			# img[img > 65536] = 65534
		if bits == '8':
			img = (255 * (img - norm_stat_min) / norm_stat_ptp * ptp_scale_factor)
			img[img < 0] = 0
			img[img > 255] = 255
			img = img.astype('uint8')

	img = np.multiply(img, alpha_blurr_32bit).astype(f'uint{bits}')
	compression = True
	if not compression:
		tifffile.imwrite(os.path.join(out_dir_recon_bits, os.path.basename(image_path)), img.astype(f'uint{bits}'),
						compression=0)
	else:
		tifffile.imwrite(os.path.join(out_dir_recon_bits, os.path.basename(image_path)), img.astype(f'uint{bits}'),
						compression='zstd')
	print(f'processed {image_path}')


def create_circular_mask(h, w, center=None, radius=None):
	if center is None:
		# use the middle of the image
		center = (int(h / 2), int(w / 2))
	if radius is None:
		# use the smallest distance between the center and image walls
		radius = min(center[0], center[1], h - center[0], w - center[1])

	Y, X = np.ogrid[:h, :w]
	dist_from_center = np.sqrt((Y - center[0]) ** 2 + (X - center[1]) ** 2)

	mask = dist_from_center <= radius
	return mask


try:
	current_node = os.environ['SLURM_JOB_NODELIST']
	# print(f'hpc run: {current_node}')
	is_hpc = True
except:
	# print('not running on SLURM!')
	is_hpc = False


crop_buffer_px = 100
# bits = '8'
circle_Mask_it = True
Circle_mask_radius = 6284
thresh_low = 50
final_recon_crop = np.s_[2000:4200, 1100:3350]
final_recon_crop = np.s_[:, :]
median_blur_radius = 5
kernel = np.ones((21, 21), np.uint8)

start_full_script = datetime.now()
proj_dir = os.path.dirname(sys.argv[0])
target_drive = os.path.join(proj_dir)
dir_ = os.path.join(proj_dir, '')

berk_mode = False
'''INPUT DIR SUFFIX'''
folder_you_want_to_reconstruct = 'projections'
'''setup directories'''
dir_ = os.path.join(proj_dir, folder_you_want_to_reconstruct)
'''B5_ hard link the projection directory'''
'''B5_ hard link the projection directory'''

if berk_mode:
	dir_ = os.path.join(proj_dir, f'{folder_you_want_to_reconstruct}_{exp}us')

print(os.path.normpath(dir_).split(os.sep))
unique_id = os.path.normpath(dir_).split(os.sep)[-2]
sample_name = os.path.normpath(dir_).split(os.sep)[-3]
project_name = os.path.normpath(dir_).split(os.sep)[-4]
trip_name = os.path.normpath(dir_).split(os.sep)[-5]
scan_location = os.path.normpath(dir_).split(os.sep)[-6]

unique_id_1 = sys.argv[1]
unique_id_2 = sys.argv[2]
z_overlap_1 = int(sys.argv[3])
z_overlap_2 = int(sys.argv[4])
bits = int(sys.argv[5])

out_dir_recon_s1 = os.path.join('/gpfs', 'Labs', 'Cheng', 'phenome', 'Reconstruction',
								scan_location, trip_name, project_name, sample_name, f'32bit_reconstructed_{unique_id_1}')
out_dir_recon_s2 = os.path.join('/gpfs', 'Labs', 'Cheng', 'phenome', 'Reconstruction',
								scan_location, trip_name, project_name, sample_name, f'32bit_reconstructed_{unique_id_2}')
out_dir_recon_s1_s2 = os.path.join('/gpfs', 'Labs', 'Cheng', 'phenome', 'Reconstruction', scan_location, trip_name,
								project_name, sample_name, f'{bits}bit_stitch')
out_dir_recon = os.path.join('/gpfs', 'Labs', 'Cheng', 'phenome', 'Reconstruction', scan_location, trip_name,
								project_name, sample_name)
mip_outout_dir = os.path.join('/gpfs', 'Labs', 'Cheng', 'phenome', 'Reconstruction', scan_location, trip_name,
								project_name, sample_name, 'MIP')

print(f'scan_location: {scan_location}')
print(f'trip_name: {trip_name}')
print(f'unique_id: {unique_id}')
print(f'sample_name: {sample_name}')
print(f'project_name: {project_name}')
print(f'stitch folder 1: {unique_id_1}')
print(f'stitch folder 2: {unique_id_2}')
print(f'stitch folder 1:2: {out_dir_recon_s1_s2}')
print(f'stitch folder 1:2: {z_overlap_1}')
print(f'z overlap 1: {z_overlap_1}')
print(f'z overlap 2: {z_overlap_2}')
print('output directory')
print(out_dir_recon)

Path(mip_outout_dir).mkdir(parents=True, exist_ok=True)
Path(out_dir_recon_s1_s2).mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
	threads_to_use_for_image_open = 24
	image_paths_s1_og = get_image_paths(out_dir_recon_s1)
	image_section_list_1 = np.ones(len(image_paths_s1_og))
	image_paths_s2_og = get_image_paths(out_dir_recon_s2)
	image_section_list_2 = np.ones(len(image_paths_s2_og))
	print(f'total images in first folder {len(image_paths_s1_og)}')
	print(f'total images in second folder {len(image_paths_s2_og)}')

	# stats_image_paths = image_paths_og[::10]
	# both_image_paths = image_paths_s1_og[]+image_paths_s2_og[]
	both_image_paths = image_paths_s1_og[:z_overlap_1] + image_paths_s2_og[z_overlap_2:]
	image_section_list = image_section_list_1[:z_overlap_1] + image_section_list_2[z_overlap_2:]

	both_image_paths_subset = (image_paths_s1_og + image_paths_s2_og)[::100]
	print(f'total images in both folders {len(both_image_paths)}')

	# avip_container = tifffile.imread(both_image_paths[0])
	mip_container = tifffile.imread(both_image_paths[0])
	bypass_read = True
	if not bypass_read:
		for image_path in both_image_paths_subset[1:]:
			print(f'miping image {image_path}')
			mip_container = np.max(np.asarray([mip_container, tifffile.imread(image_path)]), axis=0)
			# avip_container = np.mean(np.asarray([avip_container,tifffile.imread(image_path)]),axis = 0)
			# meanip_container = np.mean(np.asarray([minip_container,tifffile.imread(image_path)]),axis = 0)
		tifffile.imwrite(os.path.join(mip_outout_dir, 'MIP_32bit.tif'), mip_container)
	else:
		mip_container = tifffile.imread(os.path.join(mip_outout_dir, 'MIP_32bit.tif'))
		# avip_container = tifffile.imread(os.path.join(mip_outout_dir,f'{unique_id}_AVIP_32bit.tif'))
	print('mip timing: %s' % (datetime.now() - start_full_script))

	print(mip_container.shape)
	mip_container = reject_outliers(mip_container)
	print(mip_container.shape)

	tifffile.imwrite(os.path.join(mip_outout_dir, 'MIP_32bit_noOut.tif'), mip_container)

	random_image_index = 3000
	random_image = tifffile.imread(both_image_paths[3000])

	norm_stat_min = np.min(mip_container[np.nonzero(mip_container)])
	norm_stat_ptp = np.ptp(mip_container[np.nonzero(mip_container)])
	bit_8 = (255 * (random_image - norm_stat_min) / (norm_stat_ptp * .75))
	bit_8[bit_8 < 0] = 0
	bit_8[bit_8 > 255] = 255
	bit_8 = bit_8.astype('uint8')
	bit_16 = (65536 * (random_image - norm_stat_min) / norm_stat_ptp)
	bit_16[bit_16 < 0] = 0
	bit_16[bit_16 > 65535] = 65535
	bit_16 = bit_16.astype('uint16')
	tifffile.imwrite(os.path.join(mip_outout_dir, f'8bit_{random_image_index}.tif'), bit_8)
	tifffile.imwrite(os.path.join(mip_outout_dir, f'16bit_{random_image_index}.tif'), bit_16)
	tifffile.imwrite(os.path.join(mip_outout_dir, f'32bit_{random_image_index}.tif'), random_image)

	ydim, xdim = random_image.shape

	bit_8_thresh = (255 * (mip_container - norm_stat_min) / (norm_stat_ptp))
	bit_8_thresh[bit_8_thresh < 0] = 0
	bit_8_thresh[bit_8_thresh > 255] = 255
	bit_8_thresh = bit_8_thresh.astype('uint8')

	print('finding biggest contour')
	# threshold on white and invert
	if circle_Mask_it:
		circle_center = (int(ydim / 2), int(xdim / 2))
		mask = create_circular_mask(ydim, xdim, center=circle_center, radius=Circle_mask_radius)
		bit_8_thresh[~mask] = 0
		tifffile.imwrite(os.path.join(mip_outout_dir, '8bit_circle_mask.tif'), bit_8_thresh)
	# bit_8_thresh = cv2.medianBlur(bit_8_thresh, median_blur_radius)
	bit_8_thresh = cv2.GaussianBlur(bit_8_thresh, (21, 21), 0)

	tifffile.imwrite(os.path.join(mip_outout_dir, '8bit_guass_blurr.tif'), bit_8_thresh)

	thresh = cv2.inRange(bit_8_thresh, (thresh_low), (255))

	thresh = cv2.dilate(thresh, kernel, iterations=4)
	# get the largest contour
	contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]
	big_contour = max(contours, key=cv2.contourArea)

	convexHull = cv2.convexHull(big_contour)

	# print(f'this is the big contour?')
	# print(big_contour)
	x, y, w, h = cv2.boundingRect(big_contour)

	print('crop boundaries')
	print(f'{x},{y},{w},{h}')
	# draw white filled contour on black background to use as new alpha channel
	new_alpha = np.zeros_like(thresh)
	new_alpha = cv2.drawContours(new_alpha, [big_contour], 0, 255, -1)
	tifffile.imwrite(os.path.join(mip_outout_dir, '8bit_contour.tif'), new_alpha)

	new_alpha = cv2.dilate(new_alpha, kernel, iterations=1)
	alpha_blurr = cv2.GaussianBlur(new_alpha, (21, 21), 0)
	alpha_blurr_32bit = np.divide(alpha_blurr, 255, dtype=np.float32)
	alpha_blurr_32bit[thresh > 0] = 1
	masked_image_8bit = np.multiply(bit_8, alpha_blurr_32bit).astype('uint8')

	aphla_mask_path = os.path.join(mip_outout_dir, '32bit_alpha_mask.tif')
	tifffile.imwrite(aphla_mask_path, alpha_blurr_32bit)
	tifffile.imwrite(os.path.join(mip_outout_dir, f'8bit_{random_image_index}_masked.tif'), masked_image_8bit)
	tifffile.imwrite(os.path.join(mip_outout_dir, f'{random_image_index}_threshold.tif'), thresh)

	bit_conversion_timing = datetime.now()
	print(f'bit leveling to {bits} and cropping')

	filenames_chunk = np.array_split(both_image_paths, threads_to_use_for_image_open)
	image_section_list_chunk = np.array_split(image_section_list, threads_to_use_for_image_open)
	filenames_chunk_idx = np.array_split(range(0, len(both_image_paths)), threads_to_use_for_image_open)
	multi_proc_pass_list = [[idx, filenames_chunk, filenames_chunk_idx, x, y, w, h, bits, aphla_mask_path, norm_stat_min,
								norm_stat_ptp, out_dir_recon_s1_s2, image_section_list_chunk]
								for idx in range(0, threads_to_use_for_image_open)]
	# multi_proc_pass_list = [[image_paths[idx], x, y, w, h, bits, alpha_blurr] for idx in range(0, len(image_paths))]
	pool = Pool(threads_to_use_for_image_open)
	# print(multi_proc_pass_list[0])

	pool.map(bit_level_crop_stitch, multi_proc_pass_list)
	pool.close()
	pool.join()

	if False:
		exit()
	print('leveling to 8 and 16bits')

	# smear_32bit_max = cv2.medianBlur(mip_container, 5)
	# smear_32bit_min = cv2.medianBlur(minip_container, 5)
	# smear_32bit_mean = cv2.medianBlur(meanip_container, 5)
	# norm_stat_min[np.nonzero(norm_stat_min) < .00000000001] = .00000000001
	mip_container = mip_container[final_recon_crop]
	mip_container[mip_container < 0] = 0
	avip_container = avip_container[final_recon_crop]
	avip_container[avip_container < 0] = 0
	# combined_array = np.array([mip_container,minip_container])
	norm_stat_min = np.min(mip_container[np.nonzero(mip_container)])
	norm_stat_min_aip = np.min(avip_container[np.nonzero(avip_container)])
	norm_stat_mean = np.mean(mip_container[np.nonzero(mip_container)])
	norm_stat_mean_aip = np.mean(avip_container[np.nonzero(avip_container)])
	norm_stat_ptp = np.ptp(mip_container[np.nonzero(mip_container)])
	norm_stat_ptp = np.ptp(mip_container[np.nonzero(mip_container)])
	norm_stat_ptp_aip = np.ptp(avip_container[np.nonzero(avip_container)])

	print(f'norm_stat_min: {norm_stat_min}')
	print(f'norm_stat_ptp: {norm_stat_ptp}')
	print(f'mean mip: {norm_stat_mean}')

	print(f'norm_stat_min avg: {norm_stat_min_aip}')
	print(f'norm_stat_ptp avg: {norm_stat_ptp_aip}')
	print(f'mean avg: {norm_stat_mean_aip}')

	bit_8 = (255 * (mip_container - norm_stat_min) / (norm_stat_ptp * .75))
	bit_8[bit_8 < 0] = 0
	bit_8[bit_8 > 255] = 255
	bit_8 = bit_8.astype('uint8')
	bit_16 = (65536 * (mip_container - norm_stat_min) / norm_stat_ptp)
	bit_16[bit_16 < 0] = 0
	bit_16[bit_16 > 65535] = 65535
	bit_16 = bit_16.astype('uint16')

	tifffile.imwrite(os.path.join(mip_outout_dir, f'{unique_id}_MIP_8bit.tif'), bit_8)
	tifffile.imwrite(os.path.join(mip_outout_dir, f'{unique_id}_MIP_16bit.tif'), bit_16)
	tifffile.imwrite(os.path.join(mip_outout_dir, f'{unique_id}_MIP_16bit.tif'), mip_container)
	tifffile.imwrite(os.path.join(mip_outout_dir, f'{unique_id}_AVIP_32bit.tif'), avip_container)

	print('finding biggest contour')
	# threshold on white and invert
	if circle_Mask_it:
		ydim, xdim = bit_8.shape
		circle_center = (int(ydim / 2), int(xdim / 2))
		mask = create_circular_mask(ydim, xdim, center=circle_center, radius=Circle_mask_radius)
		bit_8[~mask] = 0
		tifffile.imwrite(os.path.join(mip_outout_dir, f'{unique_id}_MIP_8bit.tif'), bit_8)
	bit_8 = cv2.medianBlur(bit_8, median_blur_radius)
	thresh = cv2.inRange(bit_8, (thresh_low), (255))
	# thresh = 255 - thresh

	# get the largest contour
	contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]
	big_contour = max(contours, key=cv2.contourArea)

	# print(f'this is the big contour?')
	# print(big_contour)
	x, y, w, h = cv2.boundingRect(big_contour)

	print('crop boundaries')
	print(f'{x},{y},{w},{h}')
	# draw white filled contour on black background to use as new alpha channel
	new_alpha = np.zeros_like(thresh)
	new_alpha = cv2.drawContours(new_alpha, [big_contour], 0, 255, -1)

	kernel = np.ones((21, 21), np.uint8)
	new_alpha = cv2.dilate(new_alpha, kernel, iterations=1)
	alpha_blurr = cv2.GaussianBlur(new_alpha, (21, 21), 0)
	alpha_blurr_32bit = np.divide(alpha_blurr, 255, dtype=np.float32)
	masked_image = np.multiply(bit_16, alpha_blurr_32bit)

	mip_2 = win_scale(mip_container, norm_stat_ptp + norm_stat_mean_aip, norm_stat_ptp, f'uint{bits}', [0, 20000])
	mip_3 = win_scale(mip_container, norm_stat_ptp + norm_stat_mean_aip, norm_stat_ptp_aip, f'uint{bits}', [0, 20000])
	# mip_3 = win_scale(mip_container, .003, .001, f'uint{bits}', [0, 5000])

	# put new_alpha into alpha channel of img
	# new_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
	# new_img[:, :, 3] = new_alpha

	# # save result
	# cv2.imwrite("leaf3_alpha_cleaned.png", new_img)
	tifffile.imwrite(os.path.join(mip_outout_dir, f'{unique_id}_MIP_32bit_2.tif'), mip_2)
	tifffile.imwrite(os.path.join(mip_outout_dir, f'{unique_id}_MIP_32bit_3.tif'), mip_3)

	tifffile.imwrite(os.path.join(mip_outout_dir, f'{unique_id}_MIP_contour.tif'), new_alpha)
	tifffile.imwrite(os.path.join(mip_outout_dir, f'{unique_id}_MIP_thresh.tif'), thresh)
	tifffile.imwrite(os.path.join(mip_outout_dir, f'{unique_id}_MIP_contour_blurr_mask.tif'), alpha_blurr)
	tifffile.imwrite(os.path.join(mip_outout_dir, f'{unique_id}_MIP_16bit_masked.tif'), masked_image)

	if False:
		exit()

	bit_conversion_timing = datetime.now()
	print(f'bit leveling to {bits} and cropping')

	multi_proc_pass_list = [[image_paths_og[idx], x, y, w, h, bits, alpha_blurr] for idx in range(0, len(image_paths_og))]
	# multi_proc_pass_list = [[image_paths[idx], x, y, w, h, bits, alpha_blurr] for idx in range(0, len(image_paths))]
	pool = Pool(threads_to_use_for_image_open)

	pool = Pool(16)
	pool.map(bit_level_crop, multi_proc_pass_list)
	pool.close()
	pool.join()

	print('bit level timing: %s' % (datetime.now() - bit_conversion_timing))
	print('total script timing: %s' % (datetime.now() - start_full_script))
