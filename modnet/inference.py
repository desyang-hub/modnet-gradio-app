"""
Inference ONNX model of MODNet

Arguments:
    --image-path: path of the input image (a file)
    --output-path: path for saving the predicted alpha matte (a file)
    --model-path: path of the ONNX model

Example:
python inference_onnx.py \
    --image-path=demo.jpg --output-path=matte.png --model-path=modnet.onnx
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageOps

import onnxruntime

model_path = os.path.join(os.path.dirname(__file__), 'pretrained/modnet_photographic_portrait_matting.onnx')

def extract_object_with_transparency(img, alpha):
    """
    从输入图像中提取对象，并将非对象区域设为透明，最后裁剪到有效区域边界框大小。
    
    :param img: RGBA图像 (PIL Image对象)
    :param alpha: alpha通道 (NumPy数组)
    :return: 裁剪后的RGBA图像 (PIL Image对象)
    """
    # 创建alpha通道图像
    alpha_img = Image.fromarray(alpha)
    
    # 获取有效区域的边界框
    bbox = alpha_img.getbbox()
    if not bbox:
        return img
    
    # 裁剪图像
    cropped_img = img.crop(bbox)
    return cropped_img

def predict(im, crop=True):

    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)

    ref_size = 512

    # Get x_scale_factor & y_scale_factor to resize image
    def get_scale_factor(im_h, im_w, ref_size):
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32

        x_scale_factor = im_rw / im_w
        y_scale_factor = im_rh / im_h

        return x_scale_factor, y_scale_factor

    ##############################################
    #  Main Inference part
    ##############################################

    # 确保图像是RGB模式
    if im.mode != 'RGB':
        im = im.convert('RGB')
    
    orig_img = im.copy()
    orig_size = im.size  # (width, height)
    im_w, im_h = orig_size

    # 计算缩放因子
    x_scale_factor, y_scale_factor = get_scale_factor(im_h, im_w, ref_size)
    
    # 计算新尺寸
    new_w = int(im_w * x_scale_factor)
    new_h = int(im_h * y_scale_factor)
    
    # 调整图像大小
    resized_img = im.resize((new_w, new_h), Image.LANCZOS)
    
    # 归一化处理
    img_array = np.array(resized_img, dtype=np.float32)
    img_array = (img_array - 127.5) / 127.5
    
    # 准备输入形状 (1, 3, H, W)
    img_array = img_array.transpose(2, 0, 1)[np.newaxis, ...]
    
    # 运行模型推理
    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: img_array})
    
    # 处理输出结果
    matte = (np.squeeze(result[0]) * 255).astype(np.uint8)
    matte_img = Image.fromarray(matte).resize(orig_size, Image.LANCZOS)
    matte_array = np.array(matte_img)
    
    # 创建RGBA图像
    rgba_img = orig_img.copy()
    rgba_img.putalpha(Image.fromarray(matte_array))
    
    if crop:
        # 使用修改后的裁剪函数
        cropped_img = extract_object_with_transparency(rgba_img, matte_array)
        return cropped_img
    
    return rgba_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, required=True, help='Path to input image')
    parser.add_argument('--output-path', type=str, required=True, help='Path to output image')
    parser.add_argument('--model-path', type=str, default=model_path, help='Path to ONNX model')
    args = parser.parse_args()

    # 读取图像
    input_img = Image.open(args.image_path)
    
    # 运行预测
    result_img = predict(input_img, crop=True)
    
    # 保存结果
    result_img.save(args.output_path, 'PNG')
    print(f'Result saved to {args.output_path}')