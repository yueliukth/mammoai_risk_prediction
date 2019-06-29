
import dicom
import time
import numpy as np
from PIL import Image

# we start from a few pixels since there seems to be some artifacts in some images
start_column = 100

def process(input_path, output_path, is_left):

	dicom_content = dicom.read_file(input_path)
	img = dicom_content.pixel_array
	shape1 = img.shape
	start = time.time()

# scale pixels to [0,255]
	img = img - img.min()
	img = img*255.0/img.max()

# if image has inverted contrast
	n, bins = np.histogram(img,bins=256)
	bin_max = np.where(n == n.max())
	if bins[bin_max][0] >  256/2:
		img = 255 - img
		print "converted contrast"
        	n, bins = np.histogram(img,bins=256)
        	bin_max = np.where(n == n.max())

# get rid of machine artifacts and make sure a certain amount of low-value pixels are zero 
	if bins[bin_max][0] < 125:
		thres = bins[bin_max][0] + 10
		img[img<thres] = 0

# mirror the right posing breast
	is_flipped = False		
	if not is_left:
		img = np.fliplr(img)
		is_flipped = not is_flipped

# cut from the right
	w = img.shape[1]
	h = img.shape[0]
	
	h1 = int(h/4)
	h2 = int(h*3/4)

	windows = img[h1:h2,:]
	nums = np.sum(windows, axis=0)
	nums = np.cumsum(nums < 10)
	found = np.argmax(nums > 5)
## seems the mammogram is flipped for some reason
#	if found < 200:	
#		print 'flipping!'
#		is_flipped = not is_flipped
#		img = np.fliplr(img)
#	        windows = img[h1:h2,:]
#        	nums = np.sum(windows, axis=0)
#        	nums = np.cumsum(nums < 10)
#        	found = np.argmax(nums > 5)

	
	cutC = w if found==0 else found	
	if cutC <= start_column:
		print 'small crop1:', cutC, start_column
		return -1, -1, -1, -1, -1, -1, False
	
	img = img[:, start_column:cutC]
	w = img.shape[1]
	h = img.shape[0]

# cut from the top
	h1 = 0
	h2 = int(h/2)
	w1 = 0
	w2 = int(w/4)

	windows = img[h1:h2, w1:w2]
	nums = np.sum(windows, axis=1)
	nums = np.cumsum(nums > 50)
	found = np.argmax(nums > 40)
	
	cutR1 = found
	img = img[cutR1:,:]
	
	w = img.shape[1]
	h = img.shape[0]
# cut from the bottom
	h1 = int(h/2)
	h2 = h

	windows = img[h1:h2, w1:w2]
	nums = np.sum(windows, axis=1)
	nums = np.cumsum(nums < 50)
	found = np.argmax(nums > 40)

	cutR2 = h if found == 0 else found+h1-1
	img = img[:cutR2, :]
	
	w = img.shape[1]
	h = img.shape[0]
	
	if w < 100 or h < 100:
		print 'small crop2:', h, w
		return -1, -1, -1, -1, -1, -1, False

# convert image to RGB
	pil_img = Image.fromarray(img)
	pil_img = pil_img.convert('RGB');
	pil_img.save(output_path);
	shape2 = img.shape
	return start_column, cutC, cutR1, cutR2, shape1, shape2, is_flipped
