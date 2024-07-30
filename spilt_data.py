import os
import shutil
from loguru import logger


def split_images(dataset_dir, output_dir, num_splits):
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                # 获取图片标记名
                root = root.replace('\\', '/')
                usage = root.split('/')[-2]
                label = root.split('/')[-1]
                # 计算图片所属分组
                split = hash(file) % num_splits
                # 创建输出目录
                split_dir = os.path.join(output_dir, str(split), usage, label)
                os.makedirs(split_dir, exist_ok=True)
                # 复制图片到输出目录
                shutil.copy(os.path.join(root, file), split_dir)


def check_folder_count(folder_path, target_count):
    # 获取目录下的子文件夹列表
    subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]
    # 检查子文件夹数量是否与目标数量一致
    if len(subfolders) == target_count:
        logger.info(f'目录 {folder_path} 下有 {target_count} 个子文件夹，与目标数量一致。')
        return True
    else:
        logger.warning(f'目录 {folder_path} 下有 {len(subfolders)} 个子文件夹，与目标数量 {target_count} 不一致。')
        return False


def remove_files_in_dir(folder_path: str):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 递归删除子文件夹及其中文件
        except Exception as e:
            logger.error(f"Failed to delete {file_path}. Reason: {e}")


def auto_generate(input_dir, output_dir, count, force=False):
    if force or not check_folder_count(output_dir, count)   :
        remove_files_in_dir(output_dir)
        split_images(input_dir, output_dir, count)
        logger.info(f'已将 {input_dir} 下的图片分割到 {output_dir} 下。')
    else:
        logger.info(f'目录 {output_dir} 下已有 {count} 个子文件夹，无需分割。')


if __name__ == '__main__':
    auto_generate('./raw_dataset', './_dataset', 30, force=True)
