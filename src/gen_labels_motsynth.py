import os.path as osp
import os
import numpy as np


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


seq_root = '/media/workspace/lvweiyi/data/MOTSynth/mot_annotations_with_id/'
label_root = '/media/workspace/lvweiyi/data/MOTSynth/labels_with_modelids/'
mkdirs(label_root)
seqs = [s for s in os.listdir(seq_root)]
# seqs_1 = [s for s in os.listdir(label_root)]
# print(seqs)
# print(seqs_1)

tid_curr = 0
tid_last = -1
n = 0
for seq in seqs:
    n = n + 1
    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nweather')])

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)

    for fid, modelid, x, y, w, h, mark, label, vis, _, _, _ in gt:
        if mark == 0 or not label == 1:
            continue
        fid = int(fid)
        modelid = int(modelid)
        tid = int(modelid)
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid
        x += w / 2
        y += h / 2
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            modelid, x / seq_width, y / seq_height, w / seq_width, h / seq_height, vis)
        print(n)
        with open(label_fpath, 'a') as f:
            f.write(label_str)
