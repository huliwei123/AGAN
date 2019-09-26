import os,gzip
import numpy as np
import imageio
'''check the dir path'''
def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

'''load datasets'''
def load_mnist(dataset_path):
    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data
    data = extract_data(dataset_path, 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    return trX / 255.
'''save images'''
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)
def inverse_transform(images):
    return (images+1.)/2.
def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return imageio.imwrite(path,image)
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')
