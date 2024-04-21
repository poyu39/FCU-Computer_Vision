import os
import re
from PIL import Image
import numpy as np
import cv2

class PS:
    def __init__(self, file_path, file_prefix):
        self.file_path = file_path
        self.bmp_files = [f for f in os.listdir(file_path) if f.endswith('.bmp')]
        self.light_file = os.path.join(file_path, 'light.txt')
        self.file_prefix = file_prefix
        
    
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

    def cal_albedo_normal_pq(self):
        '''
            計算 albedo 和 normal 和 p, q
        '''
        I_list = []
        albedo_list = []
        normal_list = []
        p_list = []
        q_list = []
        
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
        albedo_image = cv2.normalize(albedo, None, 0, 255, cv2.NORM_MINMAX)
        normal = np.array(normal_list).reshape(gray_bmp_height, gray_bmp_wigth, 3)
        normal_image = cv2.normalize(normal, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f'../output/{self.file_prefix}_albedo.png', albedo_image)
        cv2.imwrite(f'../output/{self.file_prefix}_normal.png', normal_image)
        self.albedo = albedo
        self.normal = normal
        
        # p, q
        p_list = []
        q_list = []
        for i in range(len(normal)):
            for j in range(len(normal[i])):
                p_list.append(normal[i][j][0] / normal[i][j][2])
                q_list.append(normal[i][j][1] / normal[i][j][2])

        # nan 轉 0
        p_list = np.nan_to_num(p_list)
        q_list = np.nan_to_num(q_list)

        # 刪除極值
        p_list = np.clip(p_list, -1, 1)
        q_list = np.clip(q_list, -1, 1)

        # 如果遇到極值，取前一個值
        for i in range(1, len(p_list)):
            if p_list[i] == 1 or p_list[i] == -1:
                p_list[i] = p_list[i - 1]
            if q_list[i] == 1 or q_list[i] == -1:
                q_list[i] = q_list[i - 1]

        self.p = np.array(p_list).reshape((gray_bmp_height, gray_bmp_wigth))
        self.q = np.array(q_list).reshape((gray_bmp_height, gray_bmp_wigth))
    
    def cal_height_map(self):
        '''
            計算 height map
        '''
        p = self.p
        q = self.q
        height_map = np.zeros_like(p)
        height_map[0, 0] = 0
        
        # 計算 height map
        # height value = previous height value + corresponding q value
        for i in range(1, height_map.shape[0]):
            h = height_map[i-1, 0] + q[i, 0]
            if np.isnan(h):
                h = 0
            height_map[i, 0] = h
            
        # height value = previous height value + corresponding p value
        for j in range(1, height_map.shape[1]):
            h = height_map[0, j-1] + p[0, j]
            if np.isnan(h):
                h = 0
            height_map[0, j] = h
        
        # 平均值
        # height value = (previous height value + corresponding q value + previous height value + corresponding p value) / 2
        for i in range(1, height_map.shape[0]):
            for j in range(1, height_map.shape[1]):
                height_map[i, j] = (height_map[i-1, j] + q[i, j] + height_map[i, j-1] + p[i, j]) / 2
        
        # 旋轉 180 度
        height_map = np.rot90(height_map, 2)
        
        self.height_map = height_map
    
    def heigh_map_to_obj(self):
        '''
            輸出 height map obj
        '''
        with open(f'../output/{self.file_prefix}_height_map.obj', 'w') as f:
            for i in range(self.height_map.shape[0]):
                for j in range(self.height_map.shape[1]):
                    if not np.isnan(self.height_map[i, j]):
                        f.write(f'v {j} {i} {self.height_map[i, j]}\n')

            for i in range(self.height_map.shape[0] - 1):
                for j in range(self.height_map.shape[1] - 1):
                    x = j + i * self.height_map.shape[1]
                    y = j + (i + 1) * self.height_map.shape[1]
                    z = j + 1 + i * self.height_map.shape[1]
                    f.write(f'f {x + 1} {y + 1} {z + 1}\n')