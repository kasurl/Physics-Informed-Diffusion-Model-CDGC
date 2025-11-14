import os
from PIL import Image


def convert_to_grayscale(input_folder, output_folder):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # 检查文件是否为图像文件
        if not os.path.isfile(input_path) or not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # 打开图像并将其转换为灰度图像+裁剪+调整大小
        image = Image.open(input_path).convert("L")
        rangle = (100, 50, 640, 440)
        image = image.crop(rangle)  # 使用Image的crop函数，从截图中再次截取我们需要的区域
        image = Image.open(input_path).resize((512, 512))
        # 构建输出文件路径
        output_path = os.path.join(output_folder, filename)

        # 保存图像
        image.save(output_path)

        print(f"Converted {filename}  and saved as {output_path}")


# 指定输入和输出文件夹路径
input_folder_path = "/past_1/val/lwir"
output_folder_path = "/home/kamarl/PycharmProjects/PID/dataset/image/val/lwir"

# 调用函数进行转换
convert_to_grayscale(input_folder_path, output_folder_path)
