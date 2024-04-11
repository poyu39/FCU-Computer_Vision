import os
import re
from PIL import Image
import numpy as np
import cv2

class PS:
    def __init__(self, file_path):
        self.file_path = file_path
        self.bmp_files = [f for f in os.listdir(file_path) if f.endswith('.bmp')]
        self.light_file = os.path.join(file_path, 'light.txt')
        
    
    def get_lights_vector(self):
        '''
            取得光源向量
        '''
        self.lights = []
        with open(self.light_file, 'r', encoding='utf8') as f:
            for line in f.readlines():
                light = re.findall(r'-?\d+', line)[1:]
                self.lights.append(light)
    
    def get_gray_bmps(self):
        '''
            取得 bmp 灰階二維陣列
        '''
        self.gray_bmps = []
        for bmp in self.bmp_files:
            # 灰階
            gray_img = Image.open(f'{self.file_path}\\{bmp}').convert('L')
            self.gray_bmps.append(np.array(gray_img))
    
    def get_light_unit_vector(self):
        '''
            光源位置 -> 光源單位向量
        '''
        self.light_unit_vectors = []
        for light in self.lights:
            # 光源單位向量 = 光源向量 / 光源向量長度(範數)
            light_unit_vector = np.array(light, dtype=np.float64) / np.linalg.norm(light)
            self.light_unit_vectors.append(light_unit_vector)

    def cal_albedo_normal(self):
        '''
            計算 albedo 和 normal
        '''
        albedo_list = []
        normal_list = []
        I_list = []
        gray_bmp_wigth = self.gray_bmps[0].shape[1]
        gray_bmp_height = self.gray_bmps[0].shape[0]
        
        # 把每張 bmp 的灰階值轉成二維陣列
        for x in range(gray_bmp_height):
            for y in range(gray_bmp_wigth):
                I_list.append([self.gray_bmps[i][x][y] for i in range(len(self.gray_bmps))])

        I_list = np.array(I_list)
        
        for I in I_list:
            # N = S_inv * I
            N = np.dot(np.linalg.pinv(self.light_unit_vectors), I)
            
            # albedo = |N|
            albedo_list.append(np.linalg.norm(N))
            
            # normal = N / |N|
            normal_list.append(N / np.linalg.norm(N))
        
        # 轉成圖片
        albedo = np.array(albedo_list).reshape(gray_bmp_height, gray_bmp_wigth)
        albedo = cv2.normalize(albedo, None, 0, 255, cv2.NORM_MINMAX)
        normal = np.array(normal_list).reshape(gray_bmp_height, gray_bmp_wigth, 3)
        normal = cv2.normalize(normal, None, 0, 255, cv2.NORM_MINMAX)
        
        cv2.imwrite('../output/albedo.png', albedo)
        cv2.imwrite('../output/normal.png', normal)
        
        self.albedo = albedo
        self.normal = normal