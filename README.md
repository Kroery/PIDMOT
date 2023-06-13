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

