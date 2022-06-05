# BboxCutMix
This is a data augmentation for object detection, using bounding boxes.

<br/>

## <div align="center">Introduction</div>
There were no-objects images during training when using Mosaic[[1]](https://arxiv.org/abs/2009.07168) because of image center position of objects. So we implemented a augmentation pasting bboxes in the empty space of a training image(BboxCutMix).
<div align="center">
<img src="https://wiki.seareale.dev/uploads/images/gallery/2022-05/OjZ2tmvojIGVUxH1-image-1654015019066.png" hspace=20/>
<img src="https://wiki.seareale.dev/uploads/images/gallery/2022-05/8cf3VBdKAA2XwkR7-image-1654015040069.png" hspace=20/>
<p>An example of BboxCutMix</p>
</div>

<div align="center">
<img src="https://wiki.seareale.dev/uploads/images/gallery/2022-05/e7or31wDTvVZXfyw-image-1654015221774.png" hspace=20 width=200px/>
  <p>The distribution of bounding box sizes</p>
</div>

<br/>
<br/>

## <div align="center">Details</div>
There is already a method using bounding boxes like copy-paste[[2]](https://arxiv.org/abs/2012.07177). But BboxCutMix has two differences from copy-paste. 
1) Make hard to find a bounding box when training using offset margin. 
2) Have full edge information(no-occluded objects).
<div align="center">
<img src="https://wiki.seareale.dev/uploads/images/gallery/2022-05/ZCQYBhN09Tj0qck0-image-1654014749943.png" hspace=20/>
    <p>Data augmentation results</p>
</div>

The detailed procedure of BboxCutMix is shown in the image below. The offset margin was added to prevent easy finding out a bounding box. And we applied data augmentations for each cropped objects. Finally, cropped objects are inserted into the training image.

<div align="center">
<img src="https://wiki.seareale.dev/uploads/images/gallery/2022-05/zHeDBz8rS9T4K1bf-image-1654017772795.png" hspace=20/>
    <p>The procedure of BboxCutMix</p>
</div>

<br/>
<br/>

## <div align="center">How to use</div>
1. run the command
```bash
$ pip install -r requirements
```

2. initialize variables and add the code in your **Dataset** class like below.
```python
from bbox_cutmix import bbox_cutmix 
from torch.utils.data import Dataset 

    class customDataset(Dataset):
        ...
        self.ims = ...        # training images (list | (#_img, H, W, C))
        self.im_files = ...   # path of traning images (list)
        self.img_size = ...   # traning images size (int)
        self.indices = ...    # range(0, #_img)
        self.labels = ...     # labels of traning data (list | (#_img, #_obj, class, x, y, w, h))
        …
        # load image
        if bbox_cutmix:
            # h, w are height and width of img
            img, labels = bbox_cutmix(self, img, labels, h, w)
        …
```

<br/>
<br/>

### References
1. Zhiwei et al, AMRNet: Chips Augmentation in Aerial Images Object Detection, https://arxiv.org/abs/2009.07168
2. Golnaz et al, Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation, https://arxiv.org/abs/2012.07177

<br/><div align="center">
by [seareale](https://github.com/seareale) | [KVL Lab](http://vl.knu.ac.kr) | [KNU](http://knu.ac.kr)
</div>
