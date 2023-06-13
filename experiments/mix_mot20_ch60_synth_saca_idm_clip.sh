cd src
python train.py mot --arch dlaSACAidm_34 --gpus 0,1,2,3 --batch_size 24 --num_workers 4 --exp_id mixall_mot20_ch60_synth_saca_idm --load_model '/home/lvweiyi/code/PIDMOT/exp/mot/crowdhuman_motsynth_saca_idm_clip/model_60.pth' --data_cfg '../src/lib/cfg/data_20.json'
cd ..