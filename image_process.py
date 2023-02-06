"""
将模拟器里截出的游戏截图转化为状态文件，减小训练成本
比如某一关卡有三种颜色，则状态文件中：
第1行为3
第2到4行为三种颜色的rgb值
之后的每一行为实际游戏中一列所有三角块的状态值，有三种颜色，则值可能为0，1，2
"""

from PIL import Image
import os
from closed_color import min_color_diff

def get_img_rgb(file_path):
    # 获取图片所有像素的rgb值
    im = Image.open(file_path)
    im.getdata()
    pixes = im.load()
    return pixes


def get_colors(nums, pixes):
    # 获取颜色的rgb值
    colors = []
    height = 1215

    base_width = 287
    for i in range(0, nums):
        width = base_width + (720 - base_width) // (2 * nums) + i * (720 - base_width) // nums
        colors.append(pixes[width,height])
    return colors

def make_config_file(file):
    # 制作游戏状态文件，获取每个三角块的颜色状态，将游戏区域划分成20列，每列15或14个三角块
    file_name = file.split('.')[0].split('_')[0]
    color_nums = int(file.split('.')[0].split('_')[1])
    with open('config/'+file_name, 'w') as f:
        f.write(str(color_nums) + '\n') # os.linesep似乎会有多余的空格
        pixes = get_img_rgb('level_image/' + file)
        colors = get_colors(color_nums, pixes)
        for color in colors:
            f.write(str(color) + '\n')
        for i in range(0, 20):
            if i % 4 == 1 or i % 4 == 2:
                columns = []
                for j in range(0, 14):
                    
                    width, height = 36 + 72 * (i//2),  41 + (1160 * j) // 14
                    color = pixes[width, height]
                    columns.append(min_color_diff(color, colors)[1]) # 获取每个三角块最接近的颜色
                f.write(str(columns) + '\n')
            else:
                columns = []
                for j in range(0, 15):
                    width, height = 18 + 72 * (i // 2), 20 + (1160 * j) // 14
                    if j == 0:
                        if i % 4 == 0:
                            width, height = 18 + 144 * (i // 4), 21
                        else:
                            width, height = 144 - 18 + 144 * (i//4), 21
                    elif j == 14:
                        if i % 4 == 0:
                            width, height = 18 + 144 * (i // 4), 1160 - 21
                        else:
                            width, height = 144 - 18 + 144 * (i//4), 1160 - 21
                    else:
                        width, height = 36 + 72 * (i//2),  (1160 * j) // 14
                    color = pixes[width, height]
                    columns.append(min_color_diff(color, colors)[1])
                if i == 19:
                    f.write(str(columns))
                else:
                    f.write(str(columns) + '\n')
        
def get_files(path):
    # 获取目录下的所有文件
    files = []
    for file in os.listdir(path):
        if not os.path.isfile(file):
            files.append(file)
    return files

def process_image_main():
    # 转换图片
    files = get_files('level_image/')
    for file in files:
        make_config_file(file)
        
if __name__ == '__main__':
    process_image_main()

    