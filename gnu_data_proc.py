#############################################################################################
#### Imports ####
import os
import glob
import cv2
import time
import sys
import matplotlib.pyplot as plt
import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
from pylab import *
from PIL import Image
from scipy import ndimage
from ctypes import *
from numpy.ctypeslib import as_array
#############################################################################################
#### Functions ####

#############################################################################################
#### Variables ####
floatMax = 1.0e5 # arbitrarily large number. 32 bit float max approx 3.4e38
refFrameSum = zeros((512,512), dtype=np.float32) # Current sum of frames
fcount = 0.0 # frame counter
bufsize = 50 # Number of buffers for synthetic frame
## Image variables ##
rows = np.int32(512)
cols = np.int32(512)
dim = np.int32(512) # number of pixels in one dimension of square image (ie #rows = #cols = dim)
cen = dim/2 # center of image
## Sub-image variables ##
## The most important of these is 'split'. The image is separated into split * split sub-images
## The next most important is pixbuf. This is the limit on the number of pixels that can be corrected in any sub-image.
## That is, if pixbuf is 10, the code can correct translations of up to 10 pixels in any direction. 
## Note that if the sub-images are too small, pixbuf might be too big
##TODO: write logic to change pixbuf based on subsize and expected jitter
split = np.int32(8) # number of sub-images along one dimension. NOTE: split > 4 often fails for 512x512 image #TODO: get 32x32 or 16x16 sub images working
#subsize = dim/split			# number of pixels in one dimension of sub-image
subsize = 122 # sub image size
singlebuf = (dim/10) - 3	# 48. previously (-1)50 
pixbuf = subsize/10			# number of pixels along border to buffer (can't correct higher displacement than this)
offset = 12 # sub image offset
#pixbuf = np.int32(10) # number of pixels along border to buffer (can't correct higher displacement than this)
subcnt = split*split 		# total number of sub-images in both dimensions
osubsize = subsize-pixbuf	# overlap subsize
updateFrame = np.int32(1) # Updates reference frame this often. That is, if 10, updates every 10th frame Often fails if >= 3
tpixbuf = pixbuf*split # total pixel buffer
## Image path variables ##
#TODO: give the option to toggle between data sets
pathTest = 'C:/Users/Tyler/Desktop/workcenter/lrf/imagesets/Tower14.16/' # Linux, 000000.tif to 013412.tif, so 0x0 to 0x3464
#pathTest = '/home/cvorg/Pictures/Chart14.19/'
#pathTest = '/home/cvorg/Pictures/Tower13.55/'
#pathTest = 'bluredtests/shapes/' # Linux, 000001.tif to 000300.tif, so 0x0 to 0x012C
#pathTest = 'images/' # Repo, 0 to 59
n = 13413 # Chart14.16
#n = 13721 # Chart14.19
#n = 13406 # Tower13.55
#n = 301 # shapes
#n = 29 # repo
## Frame count variables ##
frcnt = 0 # Total frames grabbed
bufcnt = 0 # Frames in buffer
syncnt = 0 # Total synthetic frames output
ringidx = 0 # Index of current buffer location
## ctypes variables ##
numbufs = 4 # EDT implies 4 is ideal number of buffers, 32 for fast cameras
loops = 0 	# unlike simple_sequence.py, this simple counts number of times through the loop
last_timeouts = 0
recovering_timeout = False
#############################################################################################
#### Main Loop ####
## Initialize arrays ##
# Kernel arrays
inframe = zeros((512,512), dtype=np.uint16) # Current frame
# Display arrays
origdisp = zeros((512,512), dtype=np.uint16) # Copy of inframe, use to display orignal image
# Buffer arrays
bufframes = zeros((bufsize,512,512), dtype=np.uint16) # Store 30 frame ring buffer of inframes
avgframe = zeros((512,512), dtype = np.float) # Store average of bufframes

### image stabilization arrays ###
stableframe = zeros((512,512), dtype=np.uint16) # Current stabilized frame, not cropped
comboframe = zeros((512,512), dtype=np.uint16) # Current recombined frame, not cropped
combodisp = zeros((512,512), dtype=np.uint16) # Current recombined frame, not cropped
stabledisp = zeros((512,512), dtype=np.uint16) 
referenceframe = zeros((512,512), dtype=np.uint16) # Reference frame based on current frame mod # frames to wait before updating
referencedisp = zeros((512,512), dtype=np.uint16) 
ROIsingleframe = np.zeros((512,512), dtype = np.uint16) #50 pixel border on each side
ROIsingledisp = np.zeros((416,416), dtype = np.uint16) #50 pixel border on each side (412)
ROIstableframe = np.zeros((512, 512), dtype = np.uint16) #pixel borders = buffer * number split images
ROIstabledisp = np.zeros((512, 512), dtype = np.uint16) #pixel borders = buffer * number split images

## cropped frames after image stabilization ##
synframe = zeros((512, 512), dtype=np.uint16) # Store current synthetic frame
edgeframe = zeros((512, 512), dtype=np.uint16) # Store sobel results
keep = zeros((512, 512), dtype=np.uint16) # Keep map
curmap = zeros((512, 512), dtype=np.uint16) # Store IQM of inframe
synmap = zeros((512, 512), dtype=np.uint16) # Store IQM of synthetic frame
## Cropped display arrays ##
curdisp = zeros((512,512), dtype=np.uint16) # Use to display final modified inframe
syndisp = zeros((512,512), dtype=np.uint16) # Use to display final modified synframe

#############################################################################################
#### Timing Initialization ####
#TODO: give the option to toggle timing tests
#ts_10 = ts_21 = ts_32 = ts_43 = ts_54 = ts_65 = ts_76 = ts_87 = ts_98 = ts_10_9 = ts_11_10 = ts_12_11 = ts_13_12 = ts_14_13 = ts_now_12 = ts_13_now = 0
ts_n0 = 0
#ts_0a = 0
t_frame = 0
n_since_update = 0
t_last_update = time.time()
#############################################################################################
#### Loop Images ####
while True: # Outer loop can be removed for larger files
	for x in range(1, n):
#		ta = time.time()
		newPath = pathTest + ("%06d" % x) + '.tif' # Retrieve current image path
		inframe = plt.imread(newPath) # Get new image from path, (512,512) uint16
		
		ringidx = frcnt % bufsize
		frcnt += 1 # Increase total frames grabbed
		
#		if (bufcnt < bufsize): # Fill first thirty frames
#			bufframes[bufcnt,:,:] = inframe # 
#			bufcnt += 1 # Increase buffer count
#		if (bufcnt >= bufsize): # Buffers full, cycle out oldest
#			bufframes[ringidx,:,:] = inframe # Push new frame in
#			syncnt += 1 # Output synthetic frame, 30 frame buffer filled
#			avgframe = 0
#			for idx in range(bufsize-1):
#				bufarr = bufframes[idx,:,:].astype(np.float)
#				avgframe = avgframe + bufarr / bufsize
#			referenceframe = avgframe.astype(np.uint16)
		
		origdisp = inframe # Duplicate current frame for display later
		t0 = time.time()
#############################################################################################
#### Fix local tilt ####
		# openCV phase correlation function is used to find translation. Very slow, but very accurate
		#if (imageStableMode == 0):
		#	ROIstableframe = imageStableERT(imgs, refs, ROIstableframe, dim, split, pixbuf, subsize)
		#	cl.enqueue_write_buffer(queue, k_inframe, ROIstableframe).wait() # New frame loaded
		#elif (imageStableMode == 1):
		#	ROIsingleframe = singleImageStableERT(inframe, referenceframe, ROIsingleframe, dim) #inframe
		#	cl.enqueue_write_buffer(queue, k_inframe, ROIsingleframe).wait() # New frame loaded
		#else:
		#	cl.enqueue_write_buffer(queue, k_inframe, inframe).wait() # New frame loaded
#############################################################################################
#### IQM of current frame ####
#		t1 = time.time()
		#k_event = kernels.sobelLoc(queue, grpshapesob, locshape, k_inframe, k_edgeframe, lmsob)
		#k_event.wait() # Edge detection complete, stored in edgeframe
#		t2 = time.time()
		#k_event = kernels.blurLoc(queue, grpshapeblur, locshape, k_edgeframe, k_curmap, lmblur, k_kern)
		#k_event.wait()
#		t3 = time.time()
		# *** Remove this line when not being displayed ***
		#cl.enqueue_read_buffer(queue, k_curmap, curmap).wait() # IQM stored in curmap
#		t4 = time.time()
#############################################################################################
#### IQM of snythetic frame ####
		#k_syn = get_buf(ringidx) #k_synframe_0#
#		t5 = time.time()
		#k_event = kernels.sobelLoc(queue, grpshapesob, locshape, k_syn, k_edgeframe, lmsob)
		#k_event.wait()
#		t6 = time.time()
		#k_event = kernels.blurLoc(queue, grpshapeblur, locshape, k_edgeframe, k_synmap, lmblur, k_kern)
		#k_event.wait()
#		t7 = time.time()
#############################################################################################
#### Fuse synthetic frame ####
## Make one zero kernel
		# TODO: Trying rewriting to keep instead of map
		#k_event = kernels.make_one_zero(queue, imgshape, None, k_synmap, k_curmap)
		#k_event.wait() # Synframe made into 1's and 0's, still in synframe
#		t8 = time.time()
## Averaging convolution kernel
		# TODO: What happens if we also vary the width on this kernel?
		#k_event = kernels.averageLoc(queue, grpshapeavg, locshape, k_synmap, k_keep, lmavg)
		#k_event.wait() # Keep map filled
#		t9 = time.time()
## Fuse kernel
		# TODO: verify k_inframe and k_synframe are still the original values of inframe and synframe
		#k_event = kernels.fuse(queue, imgshape, None, k_inframe, k_keep, k_syn)
		#k_event.wait() # Inframe fused into synthetic frame
#		t10 = time.time()
		#cl.enqueue_read_buffer(queue, k_syn, synframe).wait() # Final synthetic stored in synmap
#		t11 = time.time()
#		t12 = time.time()
#############################################################################################
#### Display ####
		## Shift arrays for proper display ##
		origdisp = (origdisp >> 2).astype(np.uint8)
		#avgdisp = (avgframe << 2).astype(np.uint8)
#		avgdisp = (referenceframe >> 2).astype(np.uint8)
		#curdisp = (curmap >> 2).astype(np.uint8)
		#syndisp = (synframe >> 2).astype(np.uint8)
		#if (imageStableMode == 0):
		#	ROIstabledisp = (ROIstableframe >> 2).astype(np.uint8)
			#sub_Image0 = zeros((imgs[3].size, imgs[3].size), dtype = uint16)
			#sub_Image0 = (imgs[3].astype(uint16) >> 2).astype(uint8)
		#elif (imageStableMode == 1):
		#	ROIsingledisp = (ROIsingleframe >> 2).astype(np.uint8)
		## OpenCV to display final images ##
		t_now = time.time() # Time to display
		cv2.imshow('Original', origdisp)
#		cv2.imshow('Working Average', avgdisp)
		#cv2.imshow('Working Current', curdisp)
		#cv2.imshow('Working Synthetic', syndisp)
		#if (imageStableMode == 0):
		#	cv2.imshow('Stable Recombined', ROIstabledisp)
			#cv2.imshow('SubImage 0', sub_Image0)
		#elif (imageStableMode == 1):
		#	cv2.imshow('Stable Single Frame', ROIsingledisp)
#############################################################################################
#### Update buffers ####
#		t13 = time.time()
#		for m in range(bufsize):
#			if (m != ringidx): # Not current frame, fuse partials
#				k_syn = get_buf(m)
#				k_event = kernels.sobelLoc(queue, grpshapesob, locshape, k_syn, k_edgeframe, lmsob)
#				k_event.wait()
#				k_event = kernels.blurLoc(queue, grpshapeblur, locshape, k_edgeframe, k_synmap, lmblur, k_kern)
#				k_event.wait()
#				k_event = kernels.make_one_zero(queue, imgshape, None, k_synmap, k_curmap)
#				k_event.wait()
#				k_event = kernels.averageLoc(queue, grpshapeavg, locshape, k_synmap, k_keep, lmavg)
#				k_event.wait()
#				k_event = kernels.fuse(queue, imgshape, None, k_inframe, k_keep, k_syn)
#				k_event.wait()
#				set_buf(m, k_syn)
#			else: # Set current synthetic frame to current inframe
#				set_buf(m, k_inframe)
#		t14 = time.time()
#############################################################################################
#### Timing calculations ####
		ts_n0 += t_now - t0 # Write buffer to Final frame
		#ts_10 += t1 - t0 #
		#ts_21 += t2 - t1 #
		#ts_32 += t3 - t2 #
		#ts_43 += t4 - t3 #
		#ts_54 += t5 - t4 #
		#ts_65 += t6 - t5 #
		#ts_76 += t7 - t6 #
		#ts_87 += t8 - t7 #
		#ts_98 += t9 - t8 #
		#ts_10_9 += t10 - t9 #
		#ts_11_10 += t11 - t10 #
		#ts_12_11 += t12 - t11 #
		#ts_13_12 += t13 - t12 #
		#ts_14_13 += t14 - t13 #
		#ts_now_12 += t_now - t12 #
		#ts_13_now += t13 - t_now #
		#ts_0a += t0 - ta
#		ts_16_15 += t16 - t15 #
#		ts_17_16 += t17 - t16 #
#		ts_n_17 += t_now - t18 # Display to updating buffers
		n_since_update += 1
		if t0 - t_last_update >= 1:
			print 'tframe =', 1 / ((t_now - t_last_update) / n_since_update),
			print 'tprocessing =', 1 / (ts_n0 / n_since_update)
			#print array(( ts_10, ts_21, ts_32, ts_43, ts_54, ts_65)) / n_since_update
			#print array(( ts_76, ts_87, ts_98, ts_10_9, ts_11_10, ts_12_11)) / n_since_update
			#print array(( ts_13_12, ts_now_12, ts_13_now, ts_14_13, ts_n0, ts_0a)) / n_since_update
			t_last_update = t_now
			ts_n0 = 0#ts_10 = ts_21 = ts_32 = ts_43 = ts_54 = ts_65 = ts_76 = ts_87 = ts_98 = ts_10_9 = ts_11_10 = ts_12_11 = ts_13_12 = ts_14_13 = ts_now_12 = ts_13_now = ts_0a = 0
			n_since_update = 0
#############################################################################################
#### Close ####
		key =  cv2.waitKey(1)
cv2.destroyAllWindows()
print 'Done!'
#############################################################################################
