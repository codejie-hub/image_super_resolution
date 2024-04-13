import numpy as np
import cv2
from sklearn.decomposition import MiniBatchDictionaryLearning

'''
:author: CodeJie
:time: 2024/04/09
:desc: 使用稀疏编码和字典学习进行超分辨率重建
'''
# 定义函数判断图像块是否为边缘、轮廓、角点等富有变化的区域
def is_feature_region(patch,threshold=0.5):
    # 判断标准差是否超过阈值，表示该区域有较大的像素变化
    return np.std(patch) > threshold  # 调整阈值以适应不同图像

## 评估,计算PSNR指标
def evaluate_performance(high_res_image, reconstructed_image):
    mse = np.mean((high_res_image.astype(float) - reconstructed_image.astype(float)) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse)
    print("PSNR: {:.2f} dB".format(psnr))

def calculate_psnr(original_image, reconstructed_image):
    # 确保输入图像是float类型以避免溢出
    original_image = original_image.astype(np.float64)
    reconstructed_image = reconstructed_image.astype(np.float64)
    # 计算均方误差 (MSE)
    mse = np.mean((original_image - reconstructed_image) ** 2)
    if mse == 0:
        return float('inf')  # 如果MSE是0，则PSNR有无限大的值
        print("The PSNR value is infinite")
    # 计算PSNR
    max_pixel = 255.0  # 图像像素的最大值
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    print(f"The PSNR value is {psnr} dB")


if __name__=='__main__':
    low_img_path='F://postgraduate//demos//super_resolution//dataset//benchmark//benchmark//Set5//LR_bicubic//X2//birdx2.png'
    high_img_path='F://postgraduate//demos//super_resolution//dataset//benchmark//benchmark//Set5//HR//bird.png'
    # 读取低分辨率彩色图像
    low_res_image=cv2.imread(low_img_path)
    # 读取高分辨率彩色图像
    high_res_image=cv2.imread(high_img_path)
    # 图块尺寸
    patch_size=(1,1)
    # 重叠步长
    stride=1
    # 提取低分辨率图像的图块
    low_res_patches = []
    for i in range(0, low_res_image.shape[0] - patch_size[0] + 1, stride):
        for j in range(0, low_res_image.shape[1] - patch_size[1] + 1, stride):
            patch = low_res_image[i:i + patch_size[0], j:j + patch_size[1]]
            low_res_patches.append(patch.reshape(-1))
    # 转换为NumPy数组
    low_res_patches = np.array(low_res_patches)
    # 定义字典学习器的参数
    dictionary_size = 1024
    alpha = 1
    # 创建MiniBatchDictionaryLearning对象
    dl_low_res = MiniBatchDictionaryLearning(n_components=dictionary_size, alpha=alpha)

    # 使用低分辨率图像的图块进行字典学习
    dl_low_res.fit(low_res_patches)

    # 提取测试图像的图块，并根据特征区域进行恢复
    # 提取高分辨率图像的图块
    high_res_patches = []
    for i in range(0, high_res_image.shape[0] - patch_size[0] + 1, stride):
        for j in range(0, high_res_image.shape[1] - patch_size[1] + 1, stride):
            patch = high_res_image[i:i + patch_size[0], j:j + patch_size[1]]
            if is_feature_region(patch):
                # 使用字典进行稀疏表示
                sparse_code = dl_low_res.transform(patch.reshape(1, -1))
                # 从高分辨率字典获取重构结果
                reconstructed_patch = dl_low_res.components_.T.dot(sparse_code.T).T.reshape((patch_size[0],patch_size[1],3))
            else:
                # 对于平滑区域直接用双线性插值进行拉伸
                reconstructed_patch = cv2.resize(patch, patch_size[::-1], interpolation=cv2.INTER_LINEAR)

            high_res_patches.append(reconstructed_patch)

    # 将重构的图块重新组合成图像
    reconstructed_image = np.zeros_like(high_res_image)
    index = 0
    for i in range(0, high_res_image.shape[0] - patch_size[0] + 1, stride):
        for j in range(0, high_res_image.shape[1] - patch_size[1] + 1, stride):
            patch = high_res_patches[index]
            reconstructed_image[i:i + patch_size[0], j:j + patch_size[1]] = patch
            index += 1

    # 将图像转换为8位整数类型
    reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
    #保存
    save_path= 'F://postgraduate//demos//super_resolution//src//res//sparse_learn_res//result11.png'
    cv2.imwrite(save_path,reconstructed_image)
    #
    evaluate_performance(high_res_image,reconstructed_image)

'''
:result:

## 1 PSNR: 27.22 dB --- patch_size=8 stride=1 threadhold=10

## 2 PSNR:32.35 dB --- patch_size=5 stride=1 threadhold=5

## 3 PSNR:31.77 dB --- patch_size=5 stride=1 threadhold=1

## 4 PSNR:34.36 dB --- patch_size=3 stride=1 threadhold=3  dictionary=1024

## 5 PSNR:34.33 dB --- patch_size=3 stride=1 threadhold=1  dictionary=1024 alpha=1

## 6 PSNR:40.50 dB --- patch_size=1 stride=1 threadhold=1  dictionary=1024 alpha=1

## 7 PSNR:51.00 dB --- patch_size=1 stride=1 threadhold=1  dictionary=1024 alpha=1

## 8 PSNR:39.09 dB --- patch_size=1 stride=1 threadhold=1  dictionary=1024 alpha=1

## 9 PSNR:20.50 dB --- patch_size=3 stride=1 threadhold=1  dictionary=1024 alpha=1

## 10 PSNR:26.56 dB --- patch_size=3 stride=1 threadhold=1  dictionary=1024 alpha=1

## 11 PSNR:36.52 dB --- patch_size=1 stride=1 threadhold=1  dictionary=1024 alpha=1


'''

