#!/usr/bin/env python3

from PIL import Image

in_prefix = 'images/'
out_prefix = 'boxed/'
list_of_images = None
bounding_box_dict = {}
total_num_entries = 10000

def set_data_subset(subset_name):
    global list_of_images
    if subset_name == 'train':
        list_of_images = 'images_variant_train.txt'
    elif subset_name == 'val':
        list_of_images = 'images_variant_val.txt'
    elif subset_name == 'test':
        list_of_images = 'images_variant_test.txt'
    else:
        assert False, 'unrecognized data subset name'

def populate_bounding_box_dict(dict_fname):
    fh = None
    try:
        fh = open(dict_fname, encoding='utf8')
        for lineno, line in enumerate(fh, start=1):
            (image_name, remainder) = line.split(' ',1)
            (xmin, ymin, xmax, ymax) = remainder.split()
            bounding_box_dict[image_name] = (int(xmin), int(ymin), int(xmax), int(ymax))
            #print('Image name: ' + image_name + ', line number: ' + str(lineno))
            #print(' xmin: ' + str(xmin) + ' ymin: ' + str(ymin) + ' xmax: ' + str(xmax) + \
            #      ' ymax: ' + str(ymax))
    finally:
        if fh is not None:
            fh.close()

def box_image(image_name):
    # build paths
    in_path = in_prefix + image_name + '.jpg'
    out_path = out_prefix + image_name + '.jpg'
	
    # read unaltered image
    img = Image.open(in_path)

    # get dimensions
    width = int(img.width)
    height = int(img.height)

    # define crop box. This discards copyright info at foot of image
    crop_box = (0, 0, width, height - 20)

    # get the image's bounding box.  If image does not have an
    # associated bounding box, use the crop box
    bounding_box = bounding_box_dict.get(image_name, crop_box)

    # error checking
    if bounding_box == crop_box:
        print('Image: ' + image_name + ' does not have a bounding box, using crop box instead')
    (xmin, ymin, xmax, ymax) = bounding_box
    assert xmin >= 0, 'invalid xmin'
    assert ymin >= 0, 'invalid ymin'
    assert xmax <= width, 'invalid xmax'
    assert ymax <= height, 'invalid ymax'
        
    # crop the image using the bounding box
    cropped_image = img.crop(bounding_box)

    # write cropped image
    cropped_image.save(out_path)

    # close files
    img.close()
    cropped_image.close()

#------------------------------------------------------------------------------------------
#
#	Program entry point.
#
#   First, specify which partition ('train', 'val' or 'test')
#   of the images to box.
#
#   Second, build the dict containing the bounding box
#   of the primary aircraft for each image.
#
#   Then, for each image specified in list_of_images,
#   read image from the images folder, crop it using associated
#   bounding box to obtain primary aircraft image, then
#   write image to the boxed folder.
#

# specify the data subset we are manipulating
set_data_subset('test')

# populate the bounding box dictionary
populate_bounding_box_dict('images_box.txt')

fh = None
try:
    fh = open(list_of_images, encoding='utf8')
    for lineno, line in enumerate(fh, start=1):
        (image_name, remainder) = line.split(' ',1)
        variant = remainder.strip()
        print('Line: ' + str(lineno) + ' Image name: ' + image_name + ', variant: ' + variant)
        box_image(image_name)
finally:
    if fh is not None:
        fh.close()

