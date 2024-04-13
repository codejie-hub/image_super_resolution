import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
'''
:author: CodeJie
:time: 2024/04/10
:desc: 使用线性回归进行超分辨率重建
'''

## 评估,计算PSNR指标
def evaluate_performance(high_res_image, reconstructed_image):
    mse = np.mean((high_res_image.astype(float) - reconstructed_image.astype(float)) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse)
    print("PSNR: {:.2f} dB".format(psnr))

if __name__=='__main__':
    # 1. 读取原始低分辨率图像
    low_img_path = 'F://postgraduate//demos//super_resolution//dataset//benchmark//benchmark//Set5//LR_bicubic//X2//birdx2.png'
    low_res_image = cv2.imread(low_img_path)

    # 2. 预定义参数
    scale_factor = 2  # 高分辨率图像是低分辨率图像的2倍
    block_size_lr = 3  # 低分辨率图像块的大小
    block_size_hr = block_size_lr * scale_factor  # 高分辨率图像块的大小
    stride_lr = 1  # 滑动步长，块之间有重叠

    # 3. 从高分辨率图像中提取块
    high_img_path = 'F://postgraduate//demos//super_resolution//dataset//benchmark//benchmark//Set5//HR//bird.png'
    high_res_image = cv2.imread(high_img_path)

    # 4. 提取低分辨率和高分辨率图像块作为训练数据
    lr_patches = []
    hr_patches = []
    height_lr, width_lr, _ = low_res_image.shape
    for i in range(0, height_lr - block_size_lr + 1, stride_lr):
        for j in range(0, width_lr - block_size_lr + 1, stride_lr):
            lr_patch = low_res_image[i:i + block_size_lr, j:j + block_size_lr]
            hr_patch = high_res_image[i * scale_factor:(i + block_size_lr) * scale_factor,j * scale_factor:(j + block_size_lr) * scale_factor]
            lr_patches.append(lr_patch.flatten())
            hr_patches.append(hr_patch.flatten())

    # 5. 创建和训练线性回归模型
    X = np.array(lr_patches)
    Y = np.array(hr_patches)
    model = LinearRegression().fit(X, Y)

    # 6. 重建高分辨率图像
    reconstructed_image = np.zeros_like(high_res_image, dtype=np.float64)
    weight_sum = np.zeros_like(high_res_image, dtype=np.float64) # 用于加权平均的权重图
    for i in range(0, height_lr - block_size_lr + 1, stride_lr):
        for j in range(0, width_lr - block_size_lr + 1, stride_lr):
            lr_patch = low_res_image[i:i + block_size_lr, j:j + block_size_lr].flatten()
            hr_pred = model.predict(lr_patch.reshape(1, -1))
            hr_pred = hr_pred.reshape(block_size_hr, block_size_hr, -1)
            # 确保在操作之前将hr_pred转换为float64类型
            hr_pred = hr_pred.astype(np.float64)
            # 放置预测的高分辨率块到重建图像
            start_row_hr = i * scale_factor
            start_col_hr = j * scale_factor
            value=reconstructed_image[start_row_hr:start_row_hr + block_size_hr,start_col_hr:start_col_hr + block_size_hr]
            reconstructed_image[start_row_hr:start_row_hr + block_size_hr,start_col_hr:start_col_hr + block_size_hr] = value+hr_pred
            # 用于加权平均的权重图更新
            weight_sum[start_row_hr:start_row_hr + block_size_hr, start_col_hr:start_col_hr + block_size_hr] += 1

     # 防止除以零，添加一个小的epsilon
    epsilon = 1e-6
    # 应用加权平均来合并重建的高分辨率图像块
    reconstructed_image = reconstructed_image/(weight_sum + epsilon)

    # 转换图像数据类型
    reconstructed_image = np.clip(reconstructed_image, 0, 255).astype('uint8')

    # 展示重建的高分辨率图像
    # cv2.imshow('Reconstructed High Resolution Image', reconstructed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 保存重建的高分辨率图像
    cv2.imwrite('res/local_linear_regression_res/reconstructed_high_res_image7.jpg', reconstructed_image)
    evaluate_performance(high_res_image, reconstructed_image)

'''
:return
        filenmame                        PSNR          block_size_hr      block_size_lr      stride_lr
    1. reconstructed_high_res_image2    26.47 dB            6                   3               3
    2. reconstructed_high_res_image3    25.37 dB            6                   3               3
    3. reconstructed_high_res_image4    24.37 dB            6                   3               3
    4. reconstructed_high_res_image5    25.90 dB            6                   5               1
    5. reconstructed_high_res_image6    36.39 dB            6                   3               1   
    6. reconstructed_high_res_image7    36.24 dB            6                   3               1   

'''