import os
import numpy as np
import cv2 as cv

import sys
sys.path.insert(0, '../common_utils/')

import zipfile
from zipfile import ZipFile

from noisyutils import moirebackground
from picutils import get_properbb

class SingleTestGenerator:

    def __init__(self, iterator, random_gen, noise_level, size_in, size_out):
        self.iterator = iterator
        self.imgsize = size_out
        self.mnistsize = size_in
        self.noise = noise_level
        self.randgen = random_gen

    def __iter__(self):
        return self

    def __next__(self):        
        
        randoms_here = self.randgen.rand(12)
        blanks = 1 - self.noise * randoms_here[8]*moirebackground(randoms_here[0:8], self.imgsize)
        
        labes = []
        boxes = []        
        
        a = next(self.iterator)
        
        image = a['image'].astype(np.float32)[::-1]
        labl = a['label']
        ratio = 1 + 2 * randoms_here[9]
        w, h = int(self.mnistsize*ratio), int(self.mnistsize*ratio)
        image = cv.resize(image, (w, h))
        boundbox = get_properbb(image)

        xmin = int((self.imgsize-w)*randoms_here[10])
        ymin = int(self.imgsize*randoms_here[11])

        box = np.array([xmin, ymin, 0, 0], dtype=np.int32) + boundbox

        labes.append(labl)
        boxes.append(box)
        for j in range(h):
            y = (ymin + j) % self.imgsize
            blanks[y,xmin:xmin+w,0] *= 1 - image[j,:] / 255

        return 1 - blanks, (np.array(labes, dtype=np.float32), np.array(boxes))

class SingleTrainGenerator:

    def __init__(self, iterator, random_gen, noisefiles, num_classes, noise_level, size_in, size_out):
        self.iterator = iterator
        self.imgsize = size_out
        self.mnistsize = size_in
        self.numclasses = num_classes
        self.noise = noise_level
        self.randgen = random_gen
        self.moirefiles = noisefiles
        self.shindex = np.arange(len(self.moirefiles))
        np.random.shuffle(self.shindex)
        self.count = -1
        self.numread = -1
        self.numfile = -1
        self.randoms_pool = None
        self.moiredat = None

    def __iter__(self):
        self.shindex = np.arange(len(self.moirefiles))
        np.random.shuffle(self.shindex)
        self.count = -1
        self.numread = -1
        self.numfile = -1
        self.randoms_pool = None
        self.moiredat = None
        return self

    def __next__(self):        
        
        if self.count == self.numread:
            self.numfile = (self.numfile+1) % len(self.moirefiles)
            fileix = self.shindex[self.numfile]
            filename = self.moirefiles[fileix]
            with zipfile.ZipFile(filename, 'r') as thezip:
                contents = thezip.namelist()
                with thezip.open(contents[0]) as thefile:
                    self.moiredat = np.frombuffer(thefile.read(), dtype=np.uint8).reshape((-1,500,500,1))  
            self.numread = self.moiredat.shape[0]
            self.randoms_pool = np.random.rand(self.numread,13)
            self.count = 0
        
        randoms_here = self.randoms_pool[self.count]
        
        xs = int((500-self.imgsize) * randoms_here[0])
        ys = int((500-self.imgsize) * randoms_here[1])
        flp = int(4 * randoms_here[2])
        if flp == 0:
            datapiece = self.moiredat[self.count,ys:ys+self.imgsize,xs:xs+self.imgsize]
        elif flp == 1:
            datapiece = self.moiredat[self.count,ys+self.imgsize:ys:-1,xs:xs+self.imgsize]
        elif flp == 2:
            datapiece = self.moiredat[self.count,ys:ys+self.imgsize,xs+self.imgsize:xs:-1]
        else:
            datapiece = self.moiredat[self.count,ys+self.imgsize:ys:-1,xs+self.imgsize:xs:-1]
        blanks = 1 - self.noise * randoms_here[3]*datapiece/255        
        
        toss = int(randoms_here[9] * (self.numclasses + 1))        
        labes = np.array([self.numclasses], dtype=np.int32)
        boxes = []
        
        if (toss < self.numclasses):

            a = next(self.iterator)
            image = a['image'].astype(np.float32)[::-1]
            ratio = 1 + 3 * randoms_here[10]
            w, h = int(self.mnistsize*ratio), int(self.mnistsize*ratio)
            image = cv.resize(image, (w, h))
            boundbox = get_properbb(image)

            xmin = int((self.imgsize-w)*randoms_here[11])
            ymin = int(self.imgsize*randoms_here[12])
            
            box = np.array([xmin, ymin, 0, 0], dtype=np.int32) + boundbox

            labes[0] = a['label']
            boxes.append(box)
            
            for j in range(h):
                y = (ymin + j) % self.imgsize
                blanks[y,xmin:xmin+w,0] *= 1 - image[j,:] / 255
                
        self.count += 1        
        return 1 - blanks, (labes.astype(np.float32), np.array(boxes))