import cv2

from img_replacer import ImgReplacer

if __name__ == '__main__':
    dataset_dir = './dataset'
    img_replacer = ImgReplacer(dataset_dir)
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        _, destination_img, target_img = img_replacer.read('camera')
        source_img = frame
        kps1, des1 = img_replacer.sift(source_img)
        kps2, des2 = img_replacer.sift(destination_img)
        matchs = img_replacer.match(des2, des1)
        matchs_img = img_replacer.match_img(destination_img, source_img, kps2, kps1, matchs)
        homography = img_replacer.ransac(kps2, kps1, matchs)
        # 檢查 replace_img 與 resized_target_img 的大小是否差不多
        if homography is not None:
            resized_target_img = img_replacer.resize(target_img, destination_img)
            replace_img = img_replacer.warp(resized_target_img, homography, (source_img.shape[1], source_img.shape[0]))
            result_img = img_replacer.merge(source_img, replace_img)
            cv2.imshow('result', result_img)
        else:
            cv2.imshow('result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break