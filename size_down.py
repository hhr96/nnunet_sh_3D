import os
from PIL import Image
import numpy as np


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if fname.endswith(extension)]

    if sort:
        filenames = sorted(filenames)

    return filenames


def size_it(filename,output_dir):
    image = Image.open(filename)
    size = (256,256)
    image = image.resize(size, Image.LANCZOS)
    img_arr = np.asarray(image).astype(np.float32)

    img_shape = img_arr.shape
    new = np.zeros((256, 256), np.uint8)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
                new[i][j]= img_arr[i][j]

    new_im = Image.fromarray(new)

    new_im.save(os.path.join(output_dir, os.path.splitext(os.path.basename(filename))[0] + '.png'))
    new_im.close()


if __name__ == '__main__':
    input_dir = "inputs/ICA/test/images"
    output_dir = "inputs/ICA_256/test/images"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    img_path_list = all_files_under(input_dir, extension=".png")
    for filename in img_path_list:
        print("filename", filename)
        size_it(filename, output_dir)
