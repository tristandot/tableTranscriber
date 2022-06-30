import numpy as np
from skimage import io as img_io
from skimage import transform
from skimage import util
from tqdm import tqdm
from params import *

params = BaseOptions().parser()

# ------------------------------------------------
'''
In this block : Define paths to datasets
'''

# PATH TO DATASET ON SSD
line_gt = params.tr_data_path + '/cells.txt'
line_img = params.tr_data_path + '/cells/'
line_train = params.tr_data_path + '/split/trainset.txt'
line_test = params.tr_data_path + '/split/testset.txt'
line_val1 = params.tr_data_path + '/split/validationset1.txt'
line_val2 = params.tr_data_path + '/split/validationset2.txt'

# ------------------------------------------------
'''
In this block : Data utils for training datasets structured according to the IAM dataset
'''

def gather_cell(set='train'):
    '''
    Read given dataset from path line_gt and line_img
    return: List[Tuple(str(image path), str(ground truth))]
    '''
    gtfile = line_gt
    root_path = line_img
    if set == 'train':
        print(line_train)
        data_set = np.loadtxt(line_train, dtype=str)
    elif set == 'test':
        data_set = np.loadtxt(line_test, dtype=str)
    elif set == 'val':
        data_set = np.loadtxt(line_val1, dtype=str)
    elif set == 'val2':
        data_set = np.loadtxt(line_val2, dtype=str)
    else:
        print("Cannot find this dataset. Valid values for set are 'train', 'test', 'val' or 'val2'.")
        return
    gt = []
    print("Reading dataset...")
    for line in open(gtfile):
        if not line.startswith("#"):
            info = line.strip().split()
            name = info[0]
            name_parts = name.split('-')
            pathlist = [root_path] + ['-'.join(name_parts[:i+1]) for i in range(len(name_parts))]
            line_name = pathlist[-1]
            if (line_name not in data_set):  # if the line is not properly segmented
                continue
            img_path = '/'.join(pathlist)
            transcr = ' '.join(info[1:])
            gt.append((img_path, transcr))
    print("Reading done.")
    return gt

def main_loader(set='train'):
    '''
    Store pairs of image and its ground truth text
    return: List[Tuple(nparray(image), str(ground truth text))]
    '''
    line_map = gather_cell(set)
    data = []
    for i, (img_path, transcr) in enumerate(tqdm(line_map)):
        try:
            if(os.path.isfile(img_path + '.jpg')):
                img = img_io.imread(img_path + '.jpg', as_gray=True)
            elif(os.path.isfile(img_path + '.png')):
                img = img_io.imread(img_path + '.png', as_gray=True)
            
            img = img.astype(np.float32) / 255.0
        except:
            continue
        print(transcr.replace("|", " "))
        data += [(img, transcr.replace("|", " "))]
    return data

# test the functions
if __name__ == '__main__':
    (img_path, transcr) = gather_cell('train')[0]
    img = img_io.imread(img_path + '.png')
    print(img.shape)
    print(img)

    data = main_loader(set='train')
    print("length of train set:", len(data))
    print(data[10][0].shape)

    data = main_loader(set='test')
    print("length of test set:", len(data))

    data = main_loader(set='val')
    print("length of val set:", len(data))
    print("Success")