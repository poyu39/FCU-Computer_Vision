import os
import re
from PIL import Image
import numpy as np

class Dataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.bmp_files = [f for f in os.listdir(file_path) if f.endswith('.bmp')]
        self.light_file = os.path.join(file_path, 'light.txt')
        self.lights = []
    
    def get_bmp_light(self):
        with open(self.light_file, 'r', encoding='utf8') as f:
            for line in f.readlines():
                light = re.findall(r'-?\d+', line)[1:]
                self.lights.append(light)
        
    def convert_bmp_to_array(self):
        for bmp in self.bmp_files:
            img = Image.open(f'{self.file_path}\\{bmp}')
            img_array = np.array(img)
            print(img_array)