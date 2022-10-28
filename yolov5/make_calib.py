import os
import random
import sys
from shutil import copyfile

os.makedirs('images/calib', exist_ok=True)
os.makedirs('labels/calib', exist_ok=True)

files = []
with open('train2017.txt', 'r') as f:
    files = [l[:-1].replace('./images/train2017/', '') for l in f.readlines()]
random.seed(0)
random.shuffle(files)

for f in files[:1024]:
    print(f)
    copyfile('images/train2017/' + f, 'images/calib/' + f)
    f = f.replace('.jpg', '.txt')
    try:
        copyfile('labels/train2017/' + f, 'labels/calib/' + f)
    except:
        print('No labels.')
