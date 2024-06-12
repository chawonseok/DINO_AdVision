import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path
import cv2

def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()

def load_src_dir(src_dir):
    image_files = []
    for file in sorted(os.listdir(src_dir)):
        if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg') or file.lower().endswith('.png') or file.lower().endswith('.bmp'):
            image_files.append(os.path.abspath(os.path.join(src_dir, file)))
    print("Found {} image files.".format(len(image_files)))
    return sorted(image_files)

def calculate_rgb_averages(image_paths):
    """
    여러 이미지의 RGB 채널별 평균값을 계산합니다.
    
    Args:
        image_paths (list of str): 이미지 파일들의 경로 리스트.
    
    Returns:
        tuple: RGB 채널별 평균값 (R_avg, G_avg, B_avg)
    """
    r_values, g_values, b_values = [], [], []
    print("Calculating RGB averages...")
    for image_path in tqdm(image_paths):
        image = Image.open(image_path)
        image = image.convert('RGB')
        np_image = np.array(image)
        
        r, g, b = np_image[:,:,0], np_image[:,:,1], np_image[:,:,2]
        
        r_values.append(np.mean(r))
        g_values.append(np.mean(g))
        b_values.append(np.mean(b))
    
    R_avg = np.mean(r_values)
    G_avg = np.mean(g_values)
    B_avg = np.mean(b_values)
    print("Done!")
    
    return R_avg, G_avg, B_avg

def adjust_image_colors(pil_img, r_factor, g_factor, b_factor):
    """
    이미지의 RGB 값을 주어진 비율로 조정합니다.
    
    Args:
        pil_img (PIL image): PIL 이미지 파일.
        r_factor (float): R 채널 조정 비율.
        g_factor (float): G 채널 조정 비율.
        b_factor (float): B 채널 조정 비율.
    
    Returns:
        Image: RGB 값이 조정된 이미지 객체.
    """
    image = pil_img.convert('RGB')
    np_image = np.array(image, dtype=np.float32)
    
    # 각 채널에 대해 비율을 곱함
    np_image[:,:,0] *= r_factor
    np_image[:,:,1] *= g_factor
    np_image[:,:,2] *= b_factor
    
    # 값의 범위를 [0, 255]로 클리핑
    np_image = np.clip(np_image, 0, 255).astype(np.uint8)
    
    return Image.fromarray(np_image)

def image_processing(image_paths, base_rgb, r_avg, g_avg, b_avg, processed_dir):
    print("Processing images...")
    factor = 0.9
    
    for image_path in tqdm(image_paths):
        pil_img = Image.open(image_path)
        if base_rgb[0] * (1+factor) < r_avg:
            pil_img = adjust_image_colors(pil_img, base_rgb[0] * (1+factor) / r_avg, 1, 1)
        if base_rgb[0] * (1-factor) > r_avg:
            pil_img = adjust_image_colors(pil_img, base_rgb[0] * (1-factor) / r_avg, 1, 1)
        if base_rgb[1] * (1+factor) < g_avg:
            pil_img = adjust_image_colors(pil_img, 1, base_rgb[1] * (1+factor) / g_avg, 1)
        if base_rgb[1] * (1-factor) > g_avg:
            pil_img = adjust_image_colors(pil_img, 1, base_rgb[1] * (1-factor) / g_avg, 1)
        if base_rgb[2] * (1+factor) < b_avg:
            pil_img = adjust_image_colors(pil_img, 1, 1, base_rgb[2] * (1+factor) / b_avg)
        if base_rgb[2] * (1-factor) > b_avg:
            pil_img = adjust_image_colors(pil_img, 1, 1, base_rgb[2] * (1-factor) / b_avg)
            
        pil_img.save(processed_dir / Path(image_path).name)
    print("Done!")
    
def image_pass_check(img):
    
    show_image_with_position(img, "crop_image", 500, 50, (512, 512))
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if key == ord('s'):
        return False
    return True

def mask_to_image(mask: np.ndarray, mask_values):
    out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    # print(mask_values)
    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return out        

def show_image_with_position(image, window_name, x, y, size, grid_size=0, color=(255, 0, 0)):
    """
    이미지를 지정된 위치에 표시합니다.
    
    Args:
        image (np.ndarray): 표시할 이미지.
        window_name (str): 창 이름.
        x (int): 창의 x 좌표.
        y (int): 창의 y 좌표.
    """
    # 창 생성
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 창 크기 설정
    cv2.resizeWindow(window_name, size[1], size[0])
    
    # mosaic 이미지 라인 그리기
    if grid_size:
        img_size = image.shape[:2]
        
        mask_height = img_size[0] // grid_size
        mask_width = img_size[1] // grid_size
    
        for i in range(1, grid_size):
            cv2.line(image, (0, i * mask_height), (img_size[0], i * mask_height), color, 3)
        for j in range(1, grid_size):
            cv2.line(image, (j * mask_width, 0), (j * mask_width, img_size[1]), color, 3)
        
    # 창 위치 설정
    cv2.moveWindow(window_name, x, y)
    
    # 이미지 표시
    cv2.imshow(window_name, image)
    