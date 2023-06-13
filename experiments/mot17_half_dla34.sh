cd src
python train.py mot --gpus 0,1 --exp_id mot17_half_dla34 --load_model '/home/estar/huang/FairMOT/models/ctdet_coco_dla_2x.pth' --data_cfg '../src/lib/cfg/mot17_half.json'
cd ..