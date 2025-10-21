# 这里包含处理轨迹数据所用到的函数工具：

# 第一部分：将轨迹数据，和预处理的轨迹数据读取到字典中，包含：
# read_trajectory_txt
# read_filtering_trajectory_txt

# 第二部分：判断车辆锚框是否与预设定的斑马线上下端有交集，包含：
# on_segment
# orientation
# do_intersect
# is_intersection

# 第三部分：判断行人锚框是否与预设定的斑马线区域有交集，包含：
# is_in_poly

# 第四部分：读取标签数据和检测值数据，并进行匹配，包含：
# read_label_txt
# temporal_iou
# spatial_iou

# 第五部分：裁剪图像特征并保存
# crop_images

from PIL import Image, ImageDraw
import math
import os
import numpy as np
import cv2

image_size=(1200, 1100)

# 定义透视变换矩阵M（根据之前的计算）
M = cv2.getPerspectiveTransform(
    np.float32([[130, 500], [980, 230], [300, 870], [1170, 570]]),  # 原始斑马线B的四个角点
    np.float32([[160, 340], [1140, 340], [160, 770], [1140, 770]])  # 目标斑马线A的对应点
)

def read_trajectory_txt(file_path):
    data_dict = {}
    frame_id = None

    with open(file_path, 'r') as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        # 读取帧编号
        frame_id = int(lines[i].strip())
        i += 1

        # 读取矩形框数量
        num_boxes = int(lines[i].strip())
        i += 1

        # 初始化当前帧的矩形框列表
        frame_data = []

        # 读取矩形框数据
        for j in range(num_boxes):
            box_data = list(map(float, lines[i].strip().split()))
            frame_data.append({
                'id': int(box_data[0]),
                'label': int(box_data[1]),
                'left': box_data[2],
                'top': box_data[3],
                'right': box_data[4],
                'bottom': box_data[5]
            })
            i += 1

        # 将矩形框数据添加到字典中
        data_dict[frame_id] = frame_data

    return data_dict

def read_filtering_trajectory_txt(file_path):
    data_dict = {}
    frame_id = None

    with open(file_path, 'r') as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        # 读取帧编号
        frame_id = int(lines[i].strip())
        i += 1

        # 读取矩形框数量
        num_boxes = int(lines[i].strip())
        i += 1

        # 初始化当前帧的矩形框列表
        frame_data = []

        # 读取矩形框数据
        for j in range(num_boxes):
            box_data = list(map(float, lines[i].strip().split()))
            frame_data.append({
                'id': int(box_data[0]),
                'left': box_data[1],
                'top': box_data[2],
                'right': box_data[3],
                'bottom': box_data[4]
            })
            i += 1

        # 将矩形框数据添加到字典中
        data_dict[frame_id] = frame_data

    return data_dict

def read_id_frame_count_txt(file_path):
    """
    读取包含ID和帧计数的txt文件，并将数据存储到字典中。

    :param file_path: 文件路径
    :return: 包含ID和帧计数的字典
    """
    data_dict = {}  # 初始化字典

    with open(file_path, 'r') as file:
        lines = file.readlines()  # 读取所有行

    for line in lines:
        # 移除行末的换行符，并根据空格将行数据分割为ID和总帧数
        id_str, frame_count_str = line.strip().split()
        
        # 将ID和帧计数转换为整数，并添加到字典中
        data_dict[int(id_str)] = int(frame_count_str)

    return data_dict

# 用法示例
# file_path = 'tracking_results.txt'
# result_dict = read_trajectory_txt(file_path)

# 打印结果
# for frame_id, frame_data in result_dict.items():
    # print(f"Frame {frame_id}:")
    # for box in frame_data:
        # print(f"  Box {box['id']}: Label {box['label']}, Coordinates ({box['left']}, {box['top']}, {box['right']}, {box['bottom']})")
        
def on_segment(p, q, r):
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))


def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2


def do_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False


def is_intersection(line_segment, rect_top_left, rect_bottom_right):
    p1, p2 = line_segment
    rect_top_right = (rect_bottom_right[0], rect_top_left[1])
    rect_bottom_left = (rect_top_left[0], rect_bottom_right[1])

    if (on_segment(rect_top_left, p1, rect_bottom_right) or
            on_segment(rect_top_left, p2, rect_bottom_right)):
        return True

    if do_intersect(p1, p2, rect_top_left, rect_top_right) or \
            do_intersect(p1, p2, rect_top_right, rect_bottom_right) or \
            do_intersect(p1, p2, rect_bottom_right, rect_bottom_left) or \
            do_intersect(p1, p2, rect_bottom_left, rect_top_left):
        return True

    return False

# # 示例用法
# line_segment_A = ((142.0, 202.0), (900.0, 195.0))
# line_segment_B = ((125.0, 297.0), (925.0, 288.0))
# # rectangle_top_left = (688.0, 164.0)
# # rectangle_bottom_right = (834.0, 346.0)
# rectangle_top_left = (265.0, 654.0)
# rectangle_bottom_right = (515.0, 916.0)

# result = is_intersection(line_segment_A, rectangle_top_left, rectangle_bottom_right) or \
         # is_intersection(line_segment_B, rectangle_top_left, rectangle_bottom_right)

# print(result)
    
def is_in_poly(p, poly):
    """
    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return:
    """
    px, py = p
    is_in = False
    for i, corner in enumerate(poly):
        next_i = i + 1 if i + 1 < len(poly) else 0
        x1, y1 = corner
        x2, y2 = poly[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if y1 == 0 and y2 == 0:
            if min(x1, x2) < px < max(x1, x2):
                is_in = True
                break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in
 
# # 示例用法
# if __name__ == '__main__':
    # point = [2, 3]
    # poly = [[0, 0], [7, 3], [8, 8], [5, 5]]
    # print(is_in_poly(point, poly))

def read_label_txt(file_path):
    data_dict = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        # 读取矩形框id
        box_id = int(lines[i].strip())
        i += 1

        # 读取该矩形框id出现了几次
        num_occurrences = int(lines[i].strip())
        i += 1

        # 初始化当前矩形框id的列表
        box_data_list = []

        # 读取矩形框数据
        for j in range(num_occurrences):
            box_data = list(map(float, lines[i].strip().split()))
            box_data_list.append({
                'occurrence': int(box_data[0]),
                'frame': int(box_data[1]),
                'cls': int(box_data[2]),
                'left': box_data[3],
                'top': box_data[4],
                'right': box_data[5],
                'bottom': box_data[6]
            })
            i += 1

        # 将矩形框id的数据添加到字典中
        data_dict[box_id] = box_data_list

    return data_dict    

# 时空匹配   
# 计算时间交并比tiou的函数    
def calculate_tiou(det_start, det_end, label_start, label_end):
    # 计算交集
    intersect_start = max(det_start, label_start)
    intersect_end = min(det_end, label_end)
    intersection = max(0, intersect_end - intersect_start)
    # 计算并集
    union = (det_end - det_start) + (label_end - label_start) - intersection
    return intersection / union if union > 0 else 0

# 计算空间交并比siou的函数
def calculate_siou(det_box, label_box):
    # 计算交集
    x_left = max(det_box['left'], label_box['left'])
    y_top = max(det_box['top'], label_box['top'])
    x_right = min(det_box['right'], label_box['right'])
    y_bottom = min(det_box['bottom'], label_box['bottom'])
    intersect_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    # 计算并集
    det_area = (det_box['right'] - det_box['left']) * (det_box['bottom'] - det_box['top'])
    label_area = (label_box['right'] - label_box['left']) * (label_box['bottom'] - label_box['top'])
    union_area = det_area + label_area - intersect_area
    return intersect_area / union_area if union_area > 0 else 0
    
# 裁剪图像的函数
def crop_images(source_folder, target_folder, crop_rectangle, start_frame, end_frame, frame=32):
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # 获取所有图像文件名并筛选出.jpg文件
    image_files = sorted([f for f in os.listdir(source_folder) if f.lower().endswith('.jpg')])
    
    # 计算帧采样间隔
    total_frames = end_frame - start_frame + 1
    step = math.floor(total_frames / frame)
    
    # 下采样帧编号
    sampled_frames = [start_frame + i * step for i in range(frame) if start_frame + i * step <= end_frame]
    
    # 使用下采样帧编号对文件夹中对应的jpg文件进行裁剪
    for i, frame_number in enumerate(sampled_frames):
        file_name = image_files[frame_number-1]  # 文件列表是从0开始的，而帧数是从1开始的
        file_path = os.path.join(source_folder, file_name)

        with Image.open(file_path) as img:
            # 裁剪图像
            cropped_img = img.crop(crop_rectangle)
            # 构造输出文件路径，输出文件按照1~frame的顺序命名
            output_file_name = f"{i+1:03d}.jpg"  # 生成格式化的文件名，如001.jpg, 032.jpg
            output_file_path = os.path.join(target_folder, output_file_name)
            # 保存裁剪后的图像
            cropped_img.save(output_file_path)

# 修改后的裁剪图像的函数，带有透视变换步骤
def crop_warp_images(source_folder, target_folder, crop_rectangle, start_frame, end_frame, frame=32):
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # 获取所有图像文件名并筛选出.jpg文件
    image_files = sorted([f for f in os.listdir(source_folder) if f.lower().endswith('.jpg')])
    
    # 计算帧采样间隔
    total_frames = end_frame - start_frame + 1
    step = math.floor(total_frames / frame)
    
    # 下采样帧编号
    sampled_frames = [start_frame + i * step for i in range(frame) if start_frame + i * step <= end_frame]
    
    # 使用下采样帧编号对文件夹中对应的jpg文件进行裁剪
    for i, frame_number in enumerate(sampled_frames):
        file_name = image_files[frame_number - 1]  # 文件列表是从0开始的，而帧数是从1开始的
        file_path = os.path.join(source_folder, file_name)

        with Image.open(file_path) as img:
            # 将图像转换为numpy数组，便于使用OpenCV进行透视变换
            img_array = np.array(img)

            # 使用OpenCV进行逆透视变换
            warped_img_array = cv2.warpPerspective(img_array, M, (image_size[0], image_size[1]))

            # 将透视变换后的图像转换回PIL图像格式
            warped_img = Image.fromarray(warped_img_array)

            # 裁剪图像
            cropped_img = warped_img.crop(crop_rectangle)

            # 构造输出文件路径，输出文件按照1~frame的顺序命名
            output_file_name = f"{i + 1:03d}.jpg"  # 生成格式化的文件名，如001.jpg, 032.jpg
            output_file_path = os.path.join(target_folder, output_file_name)

            # 保存裁剪后的图像
            cropped_img.save(output_file_path)


def crop_att_images(source_folder, target_folder, vehicle_id, data_dict, crop_rectangle, start_frame, end_frame, frame=32):
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # 获取所有图像文件名并筛选出.jpg文件
    image_files = sorted([f for f in os.listdir(source_folder) if f.lower().endswith('.jpg')])
    
    # 计算帧采样间隔
    total_frames = end_frame - start_frame + 1
    step = math.floor(total_frames / frame)
    
    # 下采样帧编号
    sampled_frames = [start_frame + i * step for i in range(frame) if start_frame + i * step <= end_frame]
    
    # 使用下采样帧编号对文件夹中对应的jpg文件进行裁剪
    for i, frame_number in enumerate(sampled_frames):
        file_name = image_files[frame_number-1]  # 文件列表是从0开始的，而帧数是从1开始的
        file_path = os.path.join(source_folder, file_name)

        vehicle_image = Image.new('L', image_size, 0)
        draw = ImageDraw.Draw(vehicle_image)    
        
        # 裁剪图像
        # 确保frame_id在data_dict中
        if (frame_number+1) in data_dict:
            # 遍历该帧的所有矩形框
            for box in data_dict[frame_number + 1]:
                # 找到符合条件的矩形框id
                if box['id'] == vehicle_id:
                    # 获取并打印详细信息
                    left = box['left']
                    top = box['top']
                    right = box['right']
                    bottom = box['bottom']
                    break
                else:
                    left, top, right, bottom = 0, 0, 0, 0                   
                    
        draw.rectangle([left, top, right, bottom], fill=255)  
        
        pedestrian_image = Image.open(file_path)
        # .convert('L')
        
        vehicle_array = np.array(vehicle_image)
        pedestrian_array = np.array(pedestrian_image)

        # 增加亮度
        result_array = np.clip(vehicle_array + pedestrian_array, 0, 255).astype(np.uint8)

        # 将结果数组转换回PIL图像
        result_image = Image.fromarray(result_array)
        result_image = result_image.crop(crop_rectangle)           
        # 构造输出文件路径，输出文件按照1~frame的顺序命名
        output_file_name = f"{i+1:03d}.jpg"  # 生成格式化的文件名，如001.jpg, 032.jpg
        output_file_path = os.path.join(target_folder, output_file_name)
        
        # 保存或显示结果图像
        result_image.save(output_file_path)
        

def crop_att_object_images(source_folder, target_folder, vehicle_id, data_dict, start_frame, end_frame, frame=32, crop_size=300):
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # 获取所有图像文件名并筛选出.jpg文件
    image_files = sorted([f for f in os.listdir(source_folder) if f.lower().endswith('.jpg')])
    
    # 计算帧采样间隔
    total_frames = end_frame - start_frame + 1
    step = math.floor(total_frames / frame)
    
    # 下采样帧编号
    sampled_frames = [start_frame + i * step for i in range(frame) if start_frame + i * step <= end_frame]
    
    # 使用下采样帧编号对文件夹中对应的jpg文件进行裁剪
    for i, frame_number in enumerate(sampled_frames):
        file_name = image_files[frame_number-1]  # 文件列表是从0开始的，而帧数是从1开始的
        file_path = os.path.join(source_folder, file_name)

        vehicle_image = Image.new('L', image_size, 0)
        draw = ImageDraw.Draw(vehicle_image)    
        
        # 裁剪图像
        # 确保frame_id在data_dict中
        if (frame_number+1) in data_dict:
            # 遍历该帧的所有矩形框
            for box in data_dict[frame_number + 1]:
                # 找到符合条件的矩形框id
                if box['id'] == vehicle_id:
                    # 获取并打印详细信息
                    left = box['left']
                    top = box['top']
                    right = box['right']
                    bottom = box['bottom']
                    break
                else:
                    left, top, right, bottom = 0, 0, 0, 0                    
        
        draw.rectangle([left, top, right, bottom], fill=255)  

        center_x, center_y = (left + right)/2, (top + bottom)/2
        x1 = max(0, center_x - crop_size)
        y1 = max(0, center_y - crop_size)
        x2 = min(1200, center_x + crop_size)
        y2 = min(1100, center_y + crop_size)
        crop_rectangle = (x1, y1, x2, y2)
        
        pedestrian_image = Image.open(file_path)
        # .convert('L')
        
        vehicle_array = np.array(vehicle_image)
        pedestrian_array = np.array(pedestrian_image)

        # 增加亮度
        result_array = np.clip(vehicle_array + pedestrian_array, 0, 255).astype(np.uint8)

        # 将结果数组转换回PIL图像
        result_image = Image.fromarray(result_array)
        result_image = result_image.crop(crop_rectangle)           
        # 构造输出文件路径，输出文件按照1~frame的顺序命名
        output_file_name = f"{i+1:03d}.jpg"  # 生成格式化的文件名，如001.jpg, 032.jpg
        output_file_path = os.path.join(target_folder, output_file_name)
        
        # 保存或显示结果图像
        result_image.save(output_file_path)
        

# 定义对车辆锚框进行透视变换的函数
def apply_perspective_transform_for_box(box, M):
    # 获取锚框的四个角坐标
    points = np.array([
        [box['left'], box['top']],       # 左上角
        [box['right'], box['top']],      # 右上角
        [box['left'], box['bottom']],    # 左下角
        [box['right'], box['bottom']]    # 右下角
    ], dtype='float32')

    # 扩展为4维以符合cv2.perspectiveTransform的输入格式
    points = np.array([points])

    # 对四个角点进行透视变换
    transformed_points = cv2.perspectiveTransform(points, M)[0]

    # 计算变换后点的最小外接矩形的边界
    left = np.min(transformed_points[:, 0])
    right = np.max(transformed_points[:, 0])
    top = np.min(transformed_points[:, 1])
    bottom = np.max(transformed_points[:, 1])

    return {
        'left': left,
        'top': top,
        'right': right,
        'bottom': bottom,
    }

def crop_bot_att_images(source_folder, target_folder, vehicle_id, data_dict, crop_rectangle, start_frame, end_frame, frame=32):
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 获取所有图像文件名并筛选出.jpg文件
    image_files = sorted([f for f in os.listdir(source_folder) if f.lower().endswith('.jpg')])

    # 计算帧采样间隔
    total_frames = end_frame - start_frame + 1
    step = math.floor(total_frames / frame)

    # 下采样帧编号
    sampled_frames = [start_frame + i * step for i in range(frame) if start_frame + i * step <= end_frame]

    # 使用下采样帧编号对文件夹中对应的jpg文件进行裁剪
    for i, frame_number in enumerate(sampled_frames):
        file_name = image_files[frame_number - 1]  # 文件列表是从0开始的，而帧数是从1开始的
        file_path = os.path.join(source_folder, file_name)

        vehicle_image = Image.new('L', image_size, 0)
        draw = ImageDraw.Draw(vehicle_image)

        # 裁剪图像
        if (frame_number + 1) in data_dict:
            # 遍历该帧的所有矩形框
            for box in data_dict[frame_number + 1]:
                # 找到符合条件的矩形框id
                if box['id'] == vehicle_id:
                    # 对车辆锚框进行透视变换（处理四个角点）
                    transformed_box = apply_perspective_transform_for_box(box, M)

                    # 获取新的框的坐标
                    left = transformed_box['left']
                    top = transformed_box['top']
                    right = transformed_box['right']
                    bottom = transformed_box['bottom']

                    draw.rectangle([left, top, right, bottom], fill=255)
                    break  # 找到后直接跳出循环
        else:
            left, top, right, bottom = 0, 0, 0, 0

        pedestrian_image = Image.open(file_path)
        
        vehicle_array = np.array(vehicle_image)
        pedestrian_array = np.array(pedestrian_image)

        # 增加亮度
        result_array = np.clip(vehicle_array + pedestrian_array, 0, 255).astype(np.uint8)

        # 将结果数组转换回PIL图像
        result_image = Image.fromarray(result_array)
        result_image = result_image.crop(crop_rectangle)

        # 构造输出文件路径，输出文件按照1~frame的顺序命名
        output_file_name = f"{i + 1:03d}.jpg"  # 生成格式化的文件名，如001.jpg, 032.jpg
        output_file_path = os.path.join(target_folder, output_file_name)

        # 保存或显示结果图像
        result_image.save(output_file_path)
        

def crop_keyframe_region(source_folder, target_folder, vehicle_id, data_dict, crop_rectangle, sampled_frames):
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # 获取所有图像文件名并筛选出.jpg文件
    image_files = sorted([f for f in os.listdir(source_folder) if f.lower().endswith('.jpg')])
    
    # 使用下采样帧编号对文件夹中对应的jpg文件进行裁剪
    for i, frame_number in enumerate(sampled_frames):
        file_name = image_files[frame_number-1]  # 文件列表是从0开始的，而帧数是从1开始的
        file_path = os.path.join(source_folder, file_name)

        vehicle_image = Image.new('L', image_size, 0)
        draw = ImageDraw.Draw(vehicle_image)    
        
        # 裁剪图像
        # 确保frame_id在data_dict中
        if (frame_number+1) in data_dict:
            # 遍历该帧的所有矩形框
            for box in data_dict[frame_number + 1]:
                # 找到符合条件的矩形框id
                if box['id'] == vehicle_id:
                    # 获取并打印详细信息
                    left = box['left']
                    top = box['top']
                    right = box['right']
                    bottom = box['bottom']
                    break
                else:
                    left, top, right, bottom = 0, 0, 0, 0                   
                
        draw.rectangle([left, top, right, bottom], fill=255)  
        
        pedestrian_image = Image.open(file_path)        
        vehicle_array = np.array(vehicle_image)
        pedestrian_array = np.array(pedestrian_image)

        # 增加亮度
        result_array = np.clip(vehicle_array + pedestrian_array, 0, 255).astype(np.uint8)

        # 将结果数组转换回PIL图像
        result_image = Image.fromarray(result_array)
        result_image = result_image.crop(crop_rectangle)           
        # 构造输出文件路径，输出文件按照1~frame的顺序命名
        output_file_name = f"{i+1:03d}.jpg"  # 生成格式化的文件名，如001.jpg, 032.jpg
        output_file_path = os.path.join(target_folder, output_file_name)
        
        # 保存或显示结果图像
        result_image.save(output_file_path)
