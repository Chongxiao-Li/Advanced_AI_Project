import numpy as np
import cv2
import os
import pathlib
import shutil
import glob

def ImgResizing(mask_csv_filename, ori_img_dir, resized_img_dir):
    for line in open(mask_csv_filename, 'r', encoding="gbk").readlines()[1:]:
        arrs = line.split(',')
        name = arrs[0]
        w, h = list(map(int, arrs[1].split(' ')))
        ori_img_filename = os.path.join(ori_img_dir, name)
        origin_img = cv2.imread(ori_img_filename, 0)
        resized_img = cv2.resize(origin_img, (w, h))
        resized_img_filename = os.path.join(resized_img_dir, name)
        cv2.imwrite(resized_img_filename, resized_img)

def Imgs2Gray(img_dir):
    for img_filename in glob.glob(os.path.join(img_dir, '*.png')):
        img_rgb = cv2.imread(img_filename)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(img_filename, img_gray)


def main():
    mask_csv_filename = os.path.join('RawDataset', 'train', 'mask.csv')
    ori_img_dir = os.path.join('RawDataset', 'train', 'img')
    resized_img_dir = os.path.join('Dataset', 'train', 'img')
    pathlib.Path(resized_img_dir).mkdir(parents=True, exist_ok=True)
    ImgResizing(mask_csv_filename, ori_img_dir, resized_img_dir)

    test_img_source_dir = os.path.join('RawDataset', 'test')
    test_img_target_dir = os.path.join('Dataset', 'test')
    shutil.copytree(test_img_source_dir, test_img_target_dir)

    Imgs2Gray(os.path.join(test_img_target_dir, 'img'))


if __name__ == '__main__':
    main()

