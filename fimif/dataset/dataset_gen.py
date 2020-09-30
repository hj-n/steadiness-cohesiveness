import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
import tadasets

def swiss_roll(nn, rr):
    return tadasets.swiss_roll(n=nn, r=rr)

def shpere(nn, rr):
    return tadasets.sphere(n=nn, r=rr)

def torus(nn, cc, aa):
    return tadasets.torus(n=nn, c=cc, a=aa)


# MNIST Data Loader Class
class MnistDataloader():
    def __init__(self, test_images_filepath, test_labels_filepath):
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_test, y_test)   


def mnist_test():
    # returns image / label data from mnist TEST dataset
    mnistLoader = MnistDataloader("./raw_data/mnist_test/t10k-images-idx3-ubyte", "./raw_data/mnist_test/t10k-labels-idx1-ubyte")
    return mnistLoader.load_data()
    
