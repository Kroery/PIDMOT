cd src
python train.py mot --arch dlaSACAidm_34 --exp_id crowdhuman_motsynth_saca_idm_clip --gpus 0,1,2,3 --batch_size 24 --load_model '/home/lvweiyi/code/PIDMOT/exp/mot/motsynth_saca_idm_clip/model_60.pth' --num_epochs 100 --lr_step '50,80' --data_cfg '../src/lib/cfg/crowdhuman.json'
cd ..