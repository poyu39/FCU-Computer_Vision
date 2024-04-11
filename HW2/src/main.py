# Photometric Stereo

from photometric_stereo import PS

if __name__ == '__main__':
    bunny = PS('../test_datasets/bunny/')
    teapot = PS('../test_datasets/teapot/')
    
    # bunny
    # 光源單位向量
    bunny.get_lights_vector()
    bunny.get_gray_bmps()
    bunny.get_light_unit_vector()
    # 計算 albedo 和 normal
    bunny.cal_albedo_normal()