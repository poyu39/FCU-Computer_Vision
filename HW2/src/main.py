from dataset import Dataset


if __name__ == '__main__':
    bunny_df = Dataset('../test_datasets/bunny/')
    teapot_df = Dataset('../test_datasets/teapot/')
    bunny_df.convert_bmp_to_array()