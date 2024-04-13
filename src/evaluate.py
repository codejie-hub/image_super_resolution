from super_resolve_sparse import calculate_psnr
import cv2


if __name__ == '__main__':
    high_img_path='F://postgraduate//demos//super_resolution//dataset//benchmark//benchmark//B100//HR//3096.png'
    reconstructed_img_path= 'F://postgraduate//demos//super_resolution//src//res//sparse_learn_res//result7.png'
    origin_img=cv2.imread(high_img_path)
    reconstructed_img=cv2.imread(reconstructed_img_path)
    psnr=calculate_psnr(origin_img,reconstructed_img)