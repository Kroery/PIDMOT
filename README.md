# PIDMOT

## Installation
PIDMOT is built upon codebase of [FairMOT](https://github.com/ifzhang/FairMOT). We use python 3.7 and pytorch >= 1.2.0

Step1. Install PIDMOT
```shell
git clone https://github.com/Kroery/PIDMOT.git
cd PIDMOT
pip3 install -r requirements.txt
```

Step2. Install DCNv2. We use [DCNv2](https://github.com/CharlesShang/DCNv2) in our backbone network and more details can be found in their repo. 

```shell
git clone https://github.com/CharlesShang/DCNv2
cd DCNv2
./make.sh
```

## Training
* Download the training data
* Change the dataset root directory 'root' in src/lib/cfg/data.json and 'data_dir' in src/lib/opts.py
* Pretrain on MOTSynth and finetuned by CrowdHuman:
```
sh experiments/motsynth_saca_idm_clip.sh
sh experiments/crowdhuman_motsynth_saca_idm_clip.sh
```
* Train on MOT17:
```
sh experiments/mix_mot17_ch60_synth_saca_idm_clip.sh
```
* Train on MOT20:
```
sh experiments/mix_mot20_ch60_synth_saca_idm_clip.sh
```

## Tracking
* Tracking on MOT17 test set:
```
cd src
python track.py mot --arch dlaSACAidm_34 --load_model $model_path$ --test_mot17 True --match_thres 0.4 --conf_thres 0.25
```
* Tracking on MOT20 test set:
```
cd src
python track.py mot --arch dlaSACAidm_34 --load_model $model_path$ --test_mot20 True --match_thres 0.4 --conf_thres 0.25
```

## Citation
~~~
@ARTICLE{10342840,
  author={Lv, Weiyi and Zhang, Ning and Zhang, Junjie and Zeng, Dan},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={One-shot Multiple Object Tracking with Robust ID Preservation}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2023.3339609}}
~~~
