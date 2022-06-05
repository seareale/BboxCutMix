# BboxCutMix(Bounding Box CutMix)
# Written by seareale
# https://github.com/seareale

import albumentations as A
import cv2
import numpy as np
import torch

IMG_SIZE = 1280
CONFIG = {
    "p": 1,  # probability of applying bbox cutmix
    "margin_min": IMG_SIZE / 40,
    "margin_max": IMG_SIZE / 20,
    "aug_vis": 0.5,  # visiblity after augmentation
    "resize_size": IMG_SIZE / 10,
}


def load_image(self, i):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    im = self.ims[i]
    if im is None:  # not cached in ram
        # remove npy load
        path = self.im_files[i]
        im = cv2.imread(path)  # BGR
        assert im is not None, "Image Not Found " + path
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(
                im,
                (int(w0 * r), int(h0 * r)),
                interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR,
            )
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    else:
        return self.ims[i].copy(), self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized


def augment(img, labels):
    album_aug = A.Compose(
        [
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.7),
            A.ColorJitter(p=0.7),
            A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.1, rotate_limit=5, p=0.7),
            A.RandomRotate90(p=0.7),
        ],
        bbox_params=A.BboxParams(
            format="coco", label_fields=["class_labels"], min_visibility=CONFIG["aug_vis"]
        ),
    )
    transformed = album_aug(image=img, bboxes=[labels[1:]], class_labels=[labels[0]])

    cropped_img = transformed["image"]
    height = cropped_img.shape[0]
    width = cropped_img.shape[1]

    if len(transformed["class_labels"]) == 0:
        return cropped_img, None, height, width

    # bboxes = [(x,y,w,h), (x,y,w,h), ... ]
    # class_labels = [ c, c, ... ]
    cropped_lb = [transformed["class_labels"][0]] + list(transformed["bboxes"][0])

    return cropped_img, cropped_lb, height, width


def get_random_object(self):
    load_idx = np.random.randint(len(self.indices))  # idx of training image for bboxcutmix
    load_img, _, (h_l, w_l) = load_image(self, load_idx)  # img, _, resized_h, resized_w
    lb = self.labels[load_idx].copy()[np.random.randint(len(self.labels[load_idx]))]

    # random offset margin : left, right, up, down
    offset_margin = np.random.randint(CONFIG["margin_min"], CONFIG["margin_max"], size=(4))

    # bbox label added offset margin
    cropped_lb = [
        max(int((lb[1] - lb[3] / 2) * w_l) - offset_margin[0], 0),
        min(int((lb[1] + lb[3] / 2) * w_l) + offset_margin[1], w_l),
        max(int((lb[2] - lb[4] / 2) * h_l) - offset_margin[2], 0),
        min(int((lb[2] + lb[4] / 2) * h_l) + offset_margin[3], h_l),
    ]

    # selected object
    cropped_img = load_img[cropped_lb[2] : cropped_lb[3], cropped_lb[0] : cropped_lb[1]].copy()

    # object label after cropped
    # class, x, y, w, h
    cropped_lb = [
        lb[0],
        int((lb[1] - lb[3] / 2) * w_l) - cropped_lb[0],
        int((lb[2] - lb[4] / 2) * h_l) - cropped_lb[2],
        int(lb[3] * w_l),
        int(lb[4] * h_l),
    ]

    cropped_img, cropped_lb, height, width = augment(cropped_img, cropped_lb)

    return cropped_img, cropped_lb, height, width


def bbox_cutmix(self, img, labels, h, w):
    xmin = int(min(labels[:, 1] - labels[:, 3] / 2) * w)
    xmax = int(max(labels[:, 1] + labels[:, 3] / 2) * w)
    ymin = int(min(labels[:, 2] - labels[:, 4] / 2) * h)
    ymax = int(max(labels[:, 2] + labels[:, 4] / 2) * h)

    area = np.array(
        [
            [0, 0, xmin, ymin],  # 1
            [xmin, 0, xmax, ymin],  # 2
            [xmax, 0, w, ymin],  # 3
            [0, ymin, xmin, ymax],  # 4
            # 5 is the smallest box including all existing boxes
            [xmax, ymin, w, ymax],  # 6
            [0, ymax, xmin, h],  # 7
            [xmin, ymax, xmax, h],  # 8
            [xmax, ymax, w, h],  # 9
        ]
    )

    input_x = (0.5 * np.random.rand(1) * (area[:, 2] - area[:, 0])).astype(int) + area[:, 0]
    input_y = (0.5 * np.random.rand(1) * (area[:, 3] - area[:, 1])).astype(int) + area[:, 1]

    for x, y, p in zip(input_x, input_y, area):
        if torch.rand(1) > CONFIG["p"]:  # probability of bboxcutmix
            continue

        # get random object
        cropped_img, cropped_lb, height, width = get_random_object(self)

        # when there is no label after augmentation, skip area
        if cropped_lb == None:
            continue

        # when cropped_img size is bigger then area
        if width >= p[2] - x or height >= p[3] - y:
            # resize threshold
            if p[2] - x > CONFIG["resize_size"] and p[3] - y > CONFIG["resize_size"]:
                scale_rate = min((p[2] - x) / width, (p[3] - y) / height)
                album_resize = A.Compose(
                    [A.Resize(int(height * scale_rate), int(width * scale_rate))],
                    bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]),
                )
                transformed = album_resize(
                    image=cropped_img,
                    bboxes=[cropped_lb[1:]],
                    class_labels=[cropped_lb[0]],
                )
                # resized crop-obj
                cropped_img = transformed["image"]
                height = cropped_img.shape[0]
                width = cropped_img.shape[1]
                cropped_lb = [transformed["class_labels"][0]] + list(transformed["bboxes"][0])
            else:
                continue

        # normailze label
        cropped_lb[1] = (x + cropped_lb[1] + cropped_lb[3] / 2) / w
        cropped_lb[2] = (y + cropped_lb[2] + cropped_lb[4] / 2) / h
        cropped_lb[3] = cropped_lb[3] / w
        cropped_lb[4] = cropped_lb[4] / h

        # paste cropped_img, add cropped_lb
        img[y : y + height, x : x + width] = cropped_img
        labels = np.vstack((labels, np.array([cropped_lb])))

    return img, labels
