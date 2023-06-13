# PIDMOT

## Installation
PIDMOT is built upon codebase of [FairMOT](https://github.com/ifzhang/FairMOT). We use python 3.7 and pytorch >= 1.2.0

### 1. Installing on the host machine
Step1. Install PIDMOT
```shell
git clone https://github.com/noahcao/OC_SORT.git
cd OC_SORT
pip3 install -r requirements.txt
python3 setup.py develop
```

Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. Others
```shell
pip3 install cython_bbox pandas xmltodict
```
