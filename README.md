# Bbox Cutmix
This is a data augmentation for object detection, using bonding box cutmix. 

<br/>

## <div align="center">Introduction</div>
There were No-object images in training when using mosaic because of image center position of objects. So we implemented a augmentation pasting bboxes in the empty space of a training image(Bbox Cumtix).
<div align="center">
<img src="https://github.com/seareale/AGC2021_object-detection/blob/main/asset/image01.png" hspace=20/>
<img src="https://github.com/seareale/AGC2021_object-detection/blob/main/asset/image02.png" hspace=20/>
<p>An example of bbox cutmix</p>
</div>

<br/>

## <div align="center">How to use</div>
1. run the command
```bash
$ pip install -r requirements
```

2. initialize the variables and add the code in your **Dataset** class like below.
```python
from bbox_cutmix import bbox_cutmix 
from torch.utils.data import Dataset 

    class customDataset(Dataset):
        ...
        self.img_files = ...   # path of traning images (list)
        self.img_size = ...   # traning images size (int) 
        self.n = ...             # number of traning images (int) 
        self.lables = ...       # labels of traning data (numpy.ndarray)
        …
        # load image
        if bbox_cutmix:
            img, labels = bbox_cutmix(self, img, labels, h, w, index)
        …
```

<br/><div align="center">
by [Seareale](https://github.com/seareale)| [KVL Lab](http://vl.knu.ac.kr) | [KNU](http://knu.ac.kr)
</div>
