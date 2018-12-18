#!/usr/bin/env python3

import os
from PIL import Image

in_prefix = 'images/'
out_prefix = 'boxed/'
bounding_box_dict = {}
total_num_entries = 10000

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

def prepare_boxed_directory():
    boxed_dir = os.path.join(os.getcwd(), out_prefix)
    if not os.path.isdir(boxed_dir):
        os.makedirs(boxed_dir)
        print('boxed dir: ' + boxed_dir + ' created');
    else:
        print('boxed dir: ' + boxed_dir + ' already exists');

def create_boxed_images(fname):
    count = 0
    fh = None
    try:
        fh = open(fname, encoding='utf8')
        for lineno, line in enumerate(fh, start=1):
            (image_name, remainder) = line.split(' ',1)
            variant = remainder.strip()
            print('Line: ' + str(lineno) + ' Image name: ' + image_name + ', variant: ' + variant)
            box_image(image_name)
            count += 1
    finally:
        if fh is not None:
            fh.close()
    return count
        
#------------------------------------------------------------------------------------------
#
#	Program entry point.
#

partition_list = ['images_variant_train.txt',
                  'images_variant_val.txt',
                  'images_variant_test.txt']

# create the boxed directory
prepare_boxed_directory()

# populate the bounding box dictionary and error check
populate_bounding_box_dict('images_box.txt')
assert len(bounding_box_dict) == total_num_entries, 'Unexpected bounding box dictionary length'

partition_list = [('train', 'images_variant_train.txt'),
                  ('val', 'images_variant_val.txt'),
                  ('test', 'images_variant_test.txt')]

# for each partition, create boxed images
total_images = 0
for value in partition_list:
    (partition_name, list_fname) = value
    print('creating boxed images for partition: ' + partition_name + ', using: ' + list_fname)
    total_images += create_boxed_images(list_fname)

# error check
assert total_images == total_num_entries, 'Unexpected number of images boxed'


