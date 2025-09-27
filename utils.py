import csv
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tqdm
from skimage import measure # 导入用于轮廓检测的库

LOGGER = logging.getLogger(__name__)


def plot_overlay_segmentation(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    image_transform=lambda x: x,
    save_depth=4,
    alpha=0.5,  # 叠加热力图的透明度
    cmap='magma',  # 热力图的颜色映射
    vmin=None,  # 颜色映射的最小值
    vmax=None,  # 颜色映射的最大值
    interpolation='bilinear',  # 热力图叠加时的插值方法
    anomaly_threshold=0.4,  # 用于显示异常区域和绘制轮廓的阈值
    contour_color='tomato',  # 异常轮廓的颜色
    contour_linewidth=2,  # 异常轮廓的线宽
):
    """生成并保存将异常分割热力图叠加在原始图像上的图片。

    Args:
        savefolder (str): 保存生成图片的目录。
        image_paths (List[str]): 原始图片的文件路径列表。
        segmentations (List[np.ndarray]): 生成的异常分割图（热力图）列表。
                                         预期为 2D (H, W) 或 3D (1, H, W) 数组。
        anomaly_scores (List[float], optional): 每张图片的异常分数。
        image_transform (function): 应用于图片的可选转换函数。
        save_depth (int): 用于生成唯一文件名的图片路径组件的数量。
        alpha (float): 叠加热力图的透明度。
        cmap (str): 热力图使用的颜色映射。
        vmin (float, optional): 颜色映射的最小值。
        vmax (float, optional): 颜色映射的最大值。
        interpolation (str): 热力图叠加时的插值方法。
        anomaly_threshold (float): 用于突出显示高异常区域并绘制轮廓的阈值。
        contour_color (str): 异常轮廓的颜色。
        contour_linewidth (int): 异常轮廓的线宽。
    """
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    for image_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Overlay Images...",
        leave=False,
    ):
        # 图像加载与转换
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform(image) # 应用传入的转换函数
        
        # 将转换后的图像转换为适合 matplotlib 显示的 NumPy 数组 (H, W, C)
        image_np = None
        if isinstance(image, np.ndarray):
            image_np = image
            if image_np.ndim == 3 and image_np.shape[0] in [1, 3] and image_np.shape[2] not in [1, 3]:
                image_np = image_np.transpose(1, 2, 0) # 从 (C, H, W) 转换为 (H, W, C)
        elif torch.is_tensor(image):
            image_np = image.cpu().numpy()
            if image_np.ndim == 3 and image_np.shape[0] in [1, 3]:
                image_np = image_np.transpose(1, 2, 0) # 从 (C, H, W) 转换为 (H, W, C)
            if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                image_np = (image_np * 255).astype(np.uint8) # 如果是浮点数，转换为 0-255 uint8
        else:
            LOGGER.warning(f"转换后不支持的图像格式: {type(image)}。尝试直接绘图。")
            image_np = image

        if image_np is None:
            LOGGER.error(f"处理图像失败: {image_path}")
            continue

        # 分割图处理：确保是 2D 数组
        if segmentation.ndim == 3 and segmentation.shape[0] == 1:
            segmentation = np.squeeze(segmentation, axis=0)
        elif segmentation.ndim == 3 and segmentation.shape[2] == 1:
            segmentation = np.squeeze(segmentation, axis=2)
        elif segmentation.ndim != 2:
            LOGGER.warning(f"意外的分割图形状: {segmentation.shape}。预期为 2D 或带单通道维度的 3D。")

        # 对分割图进行归一化 (如果尚未归一化)
        if segmentation.max() > 1.0 or segmentation.min() < 0.0:
            seg_min = segmentation.min()
            seg_max = segmentation.max()
            if seg_max - seg_min > 1e-6:
                segmentation = (segmentation - seg_min) / (seg_max - seg_min)
            else:
                segmentation = np.zeros_like(segmentation) # 如果是平坦的，则没有异常

        # 应用阈值：低于阈值的区域设置为 0，使其在热力图中不显示颜色
        thresholded_segmentation = segmentation.copy()
        thresholded_segmentation[thresholded_segmentation < anomaly_threshold] = 0

        # 生成保存文件名
        savename_parts = image_path.split(os.sep)
        savename = "_".join(savename_parts[-save_depth:])
        savename = os.path.join(savefolder, savename)
        
        # 绘图
        f, ax = plt.subplots(1, 1)
        
        # 绘制原始图像
        ax.imshow(image_np)
        
        # 叠加热力图
        im = ax.imshow(thresholded_segmentation, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax, interpolation=interpolation)
        
        # 绘制红色轮廓
        binary_mask = (segmentation >= anomaly_threshold).astype(np.uint8)
        contours = measure.find_contours(binary_mask, 0.5) # 0.5 是二值图像的等高线级别
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], color=contour_color, linewidth=contour_linewidth)

        # 设置标题，包含异常分数
        # ax.set_title(f"Original + Anomaly Map (Score: {anomaly_score:.2f})")
        ax.axis('off') # 隐藏坐标轴

        f.set_size_inches(6, 6) # 设置图片大小
        f.tight_layout() # 调整布局
        f.savefig(savename) # 保存图片
        plt.close(f) # 关闭当前图以释放内存      
        
def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
):
    """Generates and saves anomaly segmentation images.

    Args:
        savefolder (str): Directory to save the generated images.
        image_paths (List[str]): List of paths to original images.
        segmentations (List[np.ndarray]): List of generated anomaly segmentation maps.
        anomaly_scores (List[float], optional): Anomaly scores for each image. Defaults to None.
        mask_paths (List[str], optional): List of paths to ground truth masks. Defaults to None.
        image_transform (function): Optional transformation applied to images before plotting.
        mask_transform (function): Optional transformation applied to masks before plotting.
        save_depth (int): Number of path components to use for generating unique filenames.
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    for image_path, mask_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform(image)
        if not isinstance(image, np.ndarray):
            image = image.numpy()

        if masks_provided:
            if mask_path is not None:
                mask = PIL.Image.open(mask_path).convert("RGB")
                mask = mask_transform(mask)
                if not isinstance(mask, np.ndarray):
                    mask = mask.numpy()
            else:
                mask = np.zeros_like(image)
        else: # If no masks are provided, create a blank mask for consistent plotting
            mask = np.zeros_like(image)


        savename = image_path.split("/")
        savename = "_".join(savename[-save_depth:])
        savename = os.path.join(savefolder, savename)
        
        # Determine number of subplots based on mask availability
        num_plots = 2 + int(masks_provided)
        f, axes = plt.subplots(1, num_plots)
        
        axes[0].imshow(image.transpose(1, 2, 0))
        # axes[0].set_title("Original Image")
        axes[0].axis('off')

        if masks_provided:
            axes[1].imshow(mask.transpose(1, 2, 0))
            # axes[1].set_title("Ground Truth Mask")
            axes[1].axis('off')
            axes[2].imshow(segmentation)
            # axes[2].set_title(f"Anomaly Map (Score: {anomaly_score:.2f})")
            axes[2].axis('off')
        else:
            axes[1].imshow(segmentation)
            # axes[1].set_title(f"Anomaly Map (Score: {anomaly_score:.2f})")
            axes[1].axis('off')

        f.set_size_inches(3 * num_plots, 3)
        f.tight_layout()
        f.savefig(savename)
        plt.close()


def create_storage_folder(
    main_folder_path, project_folder, group_folder, run_name, mode="iterate"
):
    """Creates a unique folder path for storing experiment results.

    Args:
        main_folder_path (str): Base directory for all results.
        project_folder (str): Project-specific subfolder.
        group_folder (str): Group-specific subfolder within the project.
        run_name (str): Name for the specific run.
        mode (str): 'iterate' to create unique folder if exists, 'overwrite' to overwrite.
    
    Returns:
        str: The created storage folder path.
    """
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder, run_name)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path


def set_torch_device(gpu_ids):
    """Sets the torch device (CPU or CUDA).

    Args:
        gpu_ids (List[int]): List of GPU IDs. If empty, CPU is used.
    
    Returns:
        torch.device: The selected device.
    """
    if len(gpu_ids):
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixes random seeds for reproducibility across different libraries.

    Args:
        seed (int): The seed value.
        with_torch (bool): If True, fixes torch-related seeds.
        with_cuda (bool): If True, fixes torch+CUDA-related seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
):
    """Stores computed evaluation results as a CSV file.

    Args:
        results_path (str): Directory where the results CSV will be saved.
        results (List[List]): List of lists containing results per dataset.
                               Expected format: [[metric1, metric2, ...], ...]
        row_names (List[str], optional): Names for each row (e.g., dataset names). Defaults to None.
        column_names (List[str], optional): Names for each column (metrics).
    
    Returns:
        dict: Dictionary of mean metrics across all datasets.
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        # Handle cases where some metrics might be -1 (e.g., PRO in test mode)
        valid_results = [x[i] for x in results if x[i] != -1]
        if valid_results:
            mean_metrics[result_key] = np.mean(valid_results)
            LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))
        else:
            mean_metrics[result_key] = -1.0 # Indicate no valid results
            LOGGER.info("{0}: {1}".format(result_key, "N/A"))


    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w", newline='') as csv_file: # Added newline='' for proper CSV writing
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        
        # Write mean scores
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics