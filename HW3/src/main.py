import os
from matplotlib import pyplot as plt

from img_replacer import ImgReplacer

if __name__ == '__main__':
    
    dataset_dir = './dataset'
    img_replacer = ImgReplacer(dataset_dir)
    for i in range(1, len(os.listdir(dataset_dir))):
        print(f'正在處理 test{i}...')
        source_img, destination_img, target_img = img_replacer.read(i)
        kps1, des1 = img_replacer.sift(source_img)
        kps2, des2 = img_replacer.sift(destination_img)
        matchs = img_replacer.match(des2, des1)
        matchs_img = img_replacer.match_img(destination_img, source_img, kps2, kps1, matchs)
        homography = img_replacer.ransac(kps2, kps1, matchs)
        resized_target_img = img_replacer.resize(target_img, destination_img)
        replace_img = img_replacer.warp(resized_target_img, homography, (source_img.shape[1], source_img.shape[0]))
        result_img = img_replacer.merge(source_img, replace_img)
        img_replacer.save(matchs_img, 'match')
        img_replacer.save(result_img, 'result')