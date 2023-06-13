import os.path as osp
import os
import numpy as np


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


seq_root = '/media/workspace/lvweiyi/data/VisDrone/VisDrone2019-MOT-train/sequences/'
label_root = '/media/workspace/lvweiyi/data/VisDrone/VisDrone2019-MOT-train/labels_with_ids/'

seqs = [s for s in os.listdir(seq_root)]

for seq in seqs:
    img_path = osp.join(seq_root, seq)
    # img_path = osp.join(img_path, 'rgb')

    la_path = osp.join(label_root, seq)
    # la_path = osp.join(la_path, 'img1')

    images = [int(s.split('.')[0]) for s in os.listdir(img_path)]
    labels = [int(s.split('.')[0]) for s in os.listdir(la_path)]

    for i in images:
        if i in labels:
            continue
        else:
            print(seq, i)
            label_fpath = osp.join(la_path, '{:07d}.txt'.format(i))
            open(label_fpath, 'w')


