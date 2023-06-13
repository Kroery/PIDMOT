cd src
python train.py mot --arch dlaSACAidm_34 --exp_id motsynth_saca_idm_clip --gpus 0,1,2,3 --batch_size 48 --load_model '' --num_epochs 100 --lr_step '50,80' --data_cfg '../src/lib/cfg/motsynth.json'
cd ..