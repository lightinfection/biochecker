import os 
import concurrent.futures
import time
import numpy as np
from PIL import Image

class aug_data():

    def __init__(self, path_ctrl, path_cm):
        self.imgs_0 = [os.path.join(path_ctrl, pic)for pic in os.listdir(path_ctrl)]
        self.imgs_1 = [os.path.join(path_cm, pic) for pic in os.listdir(path_cm)]
        self.aug_imgs_0 = []
        self.aug_imgs_1 = []
        self._cropnum = 4
        self._width = 0
        self._height = 0

    def _fopen(self, im_f):
        pic = Image.open(im_f)
        if self._width!=0 and self._height!=0:
            return pic.convert("L")           
        self._width = pic.size[0]
        self._height = pic.size[1]
        return pic.convert("L")
    
    def _crop(self, im):
        item_width = (self._width/self._cropnum)
        item_height = (self._height/self._cropnum)
        box_list = []
        for i in range(self._cropnum):
            for j in range(self._cropnum):
                if i == self._cropnum-1 and j == self._cropnum-1:
                    continue
                box = (i*item_width,j*item_height,(i+1)*item_width,(j+1)*item_height)
                box_list.append(box)
        return [im.crop(box) for box in box_list]
    
    def _rotate_flip(self, im):
        operations = [Image.FLIP_TOP_BOTTOM, Image.FLIP_LEFT_RIGHT, Image.ROTATE_180]
        new_list = [im]
        for op in operations:
            new_list.append(im.transpose(op))
        return new_list
    
    def _augone(self, pic0):
        pic = self._fopen(pic0)
        buf0 = self._crop(pic)
        temp = []
        for buf in buf0:
            temp += self._rotate_flip(buf)
        return temp
    
    def _start_multi(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results_0 = executor.map(self._augone, self.imgs_0)
            results_1 = executor.map(self._augone, self.imgs_1)
        for result_0 in results_0:
            self.aug_imgs_0 += result_0
        for result_1 in results_1:
            self.aug_imgs_1 += result_1

    def _start(self):
        for i in self.imgs_0:
            x = self._augone(i)
            self.aug_imgs_0 += x
        for j in self.imgs_1:
            y = self._augone(j)
            self.aug_imgs_1 += y

class train_test_split(aug_data):

    def __init__(self, train_ratio=0.7, test_ratio=0.1, path_ctrl=None, path_cm=None):
        super().__init__(path_ctrl=path_ctrl, path_cm=path_cm)
        self.train_ratio = float(train_ratio)
        self.test_ratio = float(test_ratio)
        self._start_multi()
        label0 = [ 0 for _ in range(len(self.aug_imgs_0))]
        label1 = [ 1 for _ in range(len(self.aug_imgs_1))]
        self._total = list(zip(self.aug_imgs_0 + self.aug_imgs_1, label0 + label1))

    def _split(self):
        rng1 = np.random.default_rng(seed=14)
        rng2 = np.random.default_rng(seed=28)
        tot = len(self._total)
        train_index = list(rng1.choice(tot, size=int(self.train_ratio*tot), replace=False))
        self._train_set, temp_left = self._minus(total=self._total, tobermed=train_index)
        test_index = list(rng2.choice(len(temp_left), size=int(self.test_ratio*tot), replace=False))
        self._test_set, self._validation_set = self._minus(total=temp_left, tobermed=test_index)
        
    @staticmethod
    def _minus(total, tobermed):
        buf = []
        for i in sorted(tobermed, reverse=True):
            buf.append(total[i])
            del total[i]
        return buf, total

def main():
    test = train_test_split(path_ctrl=os.getcwd()+"/A549 PCM image dataset/Original/Ctrl",path_cm=os.getcwd()+"/A549 PCM image dataset/Original/CM")
    start = time.time()
    test._split()
    print(test._train_set[0][0])
    print("spending", float(time.time()-start), "s", len(test._train_set), len(test._validation_set), len(test._test_set))

if __name__ == "__main__":
    main()