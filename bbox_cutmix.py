# Bbox Cutmix
# Written by seareale
# https://github.com/seareale

import os

import albumentations as A
import cv2
import numpy as np
import torch

# for logging
SAVE_DIR = "./bbox_cutmix"
os.makedirs(SAVE_DIR, exist_ok=True)

CONFIG = {
    "p": 0.7,  # probability of applying bbox cutmix
    "margin_max": 20,
    "margin_min": 40,
    "aug_vis": 0.5,  # visiblity after augmentation
    "resize_thres": 10,
}


def load_image(self, i):
    path = self.img_files[i]
    im = cv2.imread(path)
    assert im is not None, "Image Not Found " + path
    h0, w0 = im.shape[:2]  # org hw
    r = self.img_size / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        im = cv2.resize(
            im,
            (int(w0 * r), int(h0 * r)),
            interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR,
        )
    return im, (h0, w0), im.shape[:2]  # im, hw_org, hw_resized


def save_image(obj, lb, name, p=None):
    h = obj.shape[0]
    w = obj.shape[1]

    img = obj.copy()
    if p is not None:
        for bbox in p:
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        if lb is not None:
            for bbox in lb:
                bbox = [
                    0,
                    (bbox[1] - bbox[3] / 2) * w,
                    (bbox[2] - bbox[4] / 2) * h,
                    bbox[3] * w,
                    bbox[4] * h,
                ]
                bbox = [int(x) for x in bbox]
                img = cv2.rectangle(
                    img, (bbox[1], bbox[2]), (bbox[1] + bbox[3], bbox[2] + bbox[4]), (0, 0, 255), 2
                )
    else:
        if lb is not None:
            lb = [int(x) for x in lb]
            img = cv2.rectangle(
                img, (lb[1], lb[2]), (lb[1] + lb[3], lb[2] + lb[4]), (0, 0, 255), 2
            )
    cv2.imwrite(f"{name}.jpg", img)


def bbox_cutmix(self, img, labels, h, w, index, cfg=CONFIG):
    # lables = [class, x,y,w,h]
    # h, w = height, width of img
    xmin = int(min(labels[:, 1] - labels[:, 3] / 2) * w)
    xmax = int(max(labels[:, 1] + labels[:, 3] / 2) * w)
    ymin = int(min(labels[:, 2] - labels[:, 4] / 2) * h)
    ymax = int(max(labels[:, 2] + labels[:, 4] / 2) * h)

    # empty grid for crop-obj
    # 1 2 3
    # 4 5 6
    # 7 8 9
    Position = [
        [0, 0, xmin, ymin],  # 1
        [xmin, 0, xmax, ymin],  # 2
        [xmax, 0, w, ymin],  # 3
        [0, ymin, xmin, ymax],  # 4
        # 5 is a bbox of all objects
        [xmax, ymin, w, ymax],  # 6
        [0, ymax, xmin, h],  # 7
        [xmin, ymax, xmax, h],  # 8
        [xmax, ymax, w, h],  # 9
    ]

    # iter empty grid
    for p in Position:
        if torch.rand(1) > CONFIG["p"]:  # probability of applying bbox cutmix
            continue
        # 1. crop-obj 생성
        load_idx = torch.randint(len(self.indices), (1,)).item()  # get a image for crop-obj
        load_img, a, (h_l, w_l) = load_image(self, load_idx)
        try:
            lb = self.labels[load_idx].copy()[
                torch.randint(len(self.labels[load_idx]), (1,)).item()
            ]
        except:
            print("-" * 30)
            print(">> bbox_cutmix - index error")
            print(f" - load index : {load_idx}")
            print(f" - label : {self.labels[load_idx]}")
            print(f" - file path : {self.img_files[load_idx]}")
            print("-" * 30)
            exit()

        xmin_load = int((lb[1] - lb[3] / 2) * w_l)
        xmax_load = int((lb[1] + lb[3] / 2) * w_l)
        ymin_load = int((lb[2] - lb[4] / 2) * h_l)
        ymax_load = int((lb[2] + lb[4] / 2) * h_l)

        # random margin for each 4 sides
        offset = torch.randint(
            low=int(self.img_size / CONFIG["margin_min"]),
            high=int(self.img_size / CONFIG["margin_max"]),
            size=(4,),
        ).numpy()
        offset_xmin = np.clip(min(xmin_load, offset[0]), 0, offset[0])
        offset_xmax = np.clip(min(w_l - xmax_load, offset[1]), 0, offset[1])
        offset_ymin = np.clip(min(ymin_load, offset[2]), 0, offset[2])
        offset_ymax = np.clip(min(h_l - ymax_load, offset[3]), 0, offset[3])

        # cropped image for the selected object
        cropped_object = load_img[
            ymin_load - offset_ymin : ymax_load + offset_ymax,
            xmin_load - offset_xmin : xmax_load + offset_xmax,
        ].copy()

        # cropped image size
        height = cropped_object.shape[0]
        width = cropped_object.shape[1]

        # label after crop
        # [class, x, y, w, h]
        cropped_label = [
            lb[0],
            offset_xmin,
            offset_ymin,
            xmax_load - xmin_load,
            ymax_load - ymin_load,
        ]

        # 2. crop-obj albumentations
        album_aug = A.Compose(  # data augmentation for crop-obj
            [
                ###
                # you can add your own augmentation
                ###
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
        try:
            transformed = album_aug(
                image=cropped_object, bboxes=[cropped_label[1:]], class_labels=[cropped_label[0]]
            )
        except:
            print("-" * 30)
            print(">> bbox_cutmix - aug error")
            print(f" - bbox coord : {cropped_label[1:]}")
            print(f" - bbox size(HxW) : {lb[4]*h_l} x {lb[3]*w_l}")
            print(f" - image size(HxW) : {height} x {width}")
            print(f" - label : {self.labels[load_idx]}")
            print(f" - file path : {self.img_files[load_idx]}")
            print("-" * 30)
            save_image(cropped_object, cropped_label, f"{SAVE_DIR}/error_{load_idx}_aug")
            continue

        # No object after augmentation
        if len(transformed["class_labels"]) == 0:
            continue

        # augmented crop-obj
        # bboxes = [(x,y,w,h), (x,y,w,h), ... ]
        # class_labels = [ c, c, ... ]
        cropped_object = transformed["image"]
        cropped_label = [transformed["class_labels"][0]] + list(transformed["bboxes"][0])
        # crop-obj size
        height = cropped_object.shape[0]
        width = cropped_object.shape[1]

        # 3. get insertion position in empty grid
        x_img = int(0.5 * torch.rand((1,)).item() * (p[2] - p[0])) + p[0]
        y_img = int(0.5 * torch.rand((1,)).item() * (p[3] - p[1])) + p[1]

        # 4. crop-obj insertion
        # when insertion not available
        if p[2] - x_img <= width or p[3] - y_img <= height:
            # resize threshold
            if (
                p[2] - x_img > self.img_size / CONFIG["resize_thres"]
                and p[3] - y_img > self.img_size / CONFIG["resize_thres"]
            ):
                scale_rate = min((p[2] - x_img) / width, (p[3] - y_img) / height)
                album_resize = A.Compose(
                    [A.Resize(int(height * scale_rate), int(width * scale_rate))],
                    bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]),
                )
                try:
                    transformed = album_resize(
                        image=cropped_object,
                        bboxes=[cropped_label[1:]],
                        class_labels=[cropped_label[0]],
                    )
                except:
                    print("-" * 30)
                    print(">> bbox_cutmix - resize error")
                    print(f" - crop coord : {cropped_label[1:]}")
                    print(f" - crop size(HxW) : {cropped_label[4]} x {cropped_label[3]}")
                    print(f" - image size(HxW) : {height} x {width}")
                    print(f" - label : {self.labels[load_idx]}")
                    print(f" - file path : {self.img_files[load_idx]}")
                    print("-" * 30)
                    save_image(
                        cropped_object, cropped_label, f"{SAVE_DIR}/error_{load_idx}_resize"
                    )
                    continue

                # resized crop-obj
                cropped_object = transformed["image"]
                cropped_label = [transformed["class_labels"][0]] + list(transformed["bboxes"][0])

                # resizd crop-obj size
                height = cropped_object.shape[0]
                width = cropped_object.shape[1]
            else:  # if over resize threshold, skip
                continue

        # normalize lables for image size
        cropped_label = [
            cropped_label[0],
            (x_img + cropped_label[1] + cropped_label[3] / 2) / w,  # (coords + margin + w/2)
            (y_img + cropped_label[2] + cropped_label[4] / 2) / h,  # (coords + margin + h/2)
            cropped_label[3] / w,
            cropped_label[4] / h,
        ]

        # add labels
        labels = np.vstack((labels, np.array([cropped_label])))

        # insert cropped image to empty grid
        try:
            img[y_img : y_img + height, x_img : x_img + width] = cropped_object
        except:
            print("-" * 30)
            print(">> bbox_cutmix - insert error")
            print(f" - crop coord : {cropped_label[1:]}")
            print(f" - crop size(HxW) : {height} x {width}")
            print(f" - min point : ({y_img}, {x_img})")
            print(f" - max point : ({y_img+height}, {x_img+width})")
            print(f" - image size(HxW) : {img.shape[1]} x {img.shape[0]}")
            print(f" - label : {self.labels[load_idx]}")
            print(f" - crop image path : {self.img_files[load_idx]}")
            print(f" - file path : {self.img_files[index]}")
            print("-" * 30)
            save_image(
                img.copy(),
                labels,
                f"{SAVE_DIR}/error_{load_idx}_{index}_insert",
                p=Position,
            )
            continue

        # bbox check
        if index in range(10, 101, 10):
            save_image(img.copy(), labels, f"{SAVE_DIR}/monitor_{index}", p=Position)

    return img, labels
