# Photometric Stereo

from photometric_stereo import PS

if __name__ == '__main__':
    bunny = PS('../test_datasets/bunny/', 'bunny')
    teapot = PS('../test_datasets/teapot/', 'teapot')
    
    # bunny
    # 光源單位向量
    bunny.get_lights_vector()
    bunny.get_gray_bmps()
    bunny.get_light_unit_vector()
    # 計算 albedo 和 normal
    bunny.cal_albedo_normal_pq()
    bunny.cal_height_map()
    bunny.heigh_map_to_obj()
    
    # teapot
    # 光源單位向量
    teapot.get_lights_vector()
    teapot.get_gray_bmps()
    teapot.get_light_unit_vector()
    # 計算 albedo 和 normal
    teapot.cal_albedo_normal_pq()
    teapot.cal_height_map()
    teapot.heigh_map_to_obj()