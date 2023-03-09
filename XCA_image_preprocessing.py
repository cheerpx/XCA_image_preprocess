import cv2
import numpy as np
class ShadowHighlight:
    """
    色阶调整
    """
    def __init__(self, image):
        self.shadows_light = 60
        img = image.astype(np.float) / 255.0
        srcR = img[:, :, 2]
        srcG = img[:, :, 1]
        srcB = img[:, :, 0]
        srcGray = 0.299 * srcR + 0.587 * srcG + 0.114 * srcB
        # 高光选区
        luminance = srcGray * srcGray
        luminance = np.where(luminance > 0.01, luminance, 0)
        # 阴影选区
        #luminance = (1 - srcGray) * (1 - srcGray)
        self.maskThreshold = np.mean(luminance)
        mask = luminance > self.maskThreshold
        imgRow = np.size(img, 0)
        imgCol = np.size(img, 1)
        print("imgRow:%d, imgCol:%d, maskThreshold:%f" % (imgRow, imgCol, self.maskThreshold))
        print("shape:", img.shape)
        self.rgbMask = np.zeros([imgRow, imgCol, 3], dtype=bool)
        self.rgbMask[:, :, 0] = self.rgbMask[:, :, 1] = self.rgbMask[:, :, 2] = mask
        self.rgbLuminance = np.zeros([imgRow, imgCol, 3], dtype=float)
        self.rgbLuminance[:, :, 0] = self.rgbLuminance[:, :, 1] = self.rgbLuminance[:, :, 2] = luminance
        self.midtonesRate = np.zeros([imgRow, imgCol, 3], dtype=float)
        self.brightnessRate = np.zeros([imgRow, imgCol, 3], dtype=float)
    def adjust_image(self, img):
        maxRate = 4
        brightness = (self.shadows_light / 100.0 - 0.0001) / maxRate
        midtones = 1 + maxRate * brightness
        self.midtonesRate[self.rgbMask] = midtones
        self.midtonesRate[~self.rgbMask] = (midtones - 1.0) / self.maskThreshold * self.rgbLuminance[
            ~self.rgbMask] + 1.0
        self.brightnessRate[self.rgbMask] = brightness
        self.brightnessRate[~self.rgbMask] = (1 / self.maskThreshold * self.rgbLuminance[~self.rgbMask]) * brightness
        outImg = 255 * np.power(img / 255.0, 1.0 / self.midtonesRate) * (1.0 / (1 - self.brightnessRate))
        img = outImg
        img[img < 0] = 0
        img[img > 255] = 255
        img = img.astype(np.uint8)
        return img
def shadow_highlight_adjust_and_save_img(psSH, origin_image):
    psSH.shadows_light = 60
    image = psSH.adjust_image(origin_image)
    cv2.imwrite('py_sh_out_03.png', image)
def hist_auto(source,out_file):
    img = cv2.imread(source, 0)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=6, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值均衡化
    dst = clahe.apply(img)
    # 使用全局直方图均衡化
    #equa = cv2.equalizeHist(img)
    # 分别显示原图，CLAHE，HE
    # cv.imshow("img", img)
    # cv2.imshow("dst", dst)
    cv2.imwrite(out_file, dst, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
source = "./2.jpg"
out_file = 'hist_auto.jpg'
hist_auto(source,out_file)
origin_image = cv2.imread(out_file)
psSH = ShadowHighlight(origin_image)
shadow_highlight_adjust_and_save_img(psSH, origin_image)

