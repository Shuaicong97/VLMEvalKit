import os
from PIL import Image
import numpy as np

def sample_frames(image_list, nframe, save_index_path=None):
    total = len(image_list)
    if nframe == -1 or nframe >= total:
        indices = list(range(total))
        sampled = image_list
    else:
        indices = np.linspace(0, total - 1, nframe, dtype=int)
        sampled = [image_list[i] for i in indices]

    if save_index_path is not None:
        os.makedirs(os.path.dirname(save_index_path), exist_ok=True)  # ✅ 加这一行

        with open(save_index_path, "w") as f:
            for idx in indices:
                f.write(f"{idx}\n")
    return sampled, indices

def compute_resize_scale(img_path, long_max=448, short_min=252):
    img = Image.open(img_path)
    w, h = img.size

    long_side = max(w, h)
    short_side = min(w, h)

    scale = 1.0

    if long_side > long_max:
        scale *= long_max / long_side
        w = int(round(w * scale))
        h = int(round(h * scale))
        long_side = max(w, h)
        short_side = min(w, h)

    if short_side < short_min:
        scale *= short_min / short_side

    return scale, scale

def resize_with_scale(img, scale):
    if scale == 1.0:
        return img
    w, h = img.size
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return img.resize((new_w, new_h), Image.BICUBIC)



def process_sequence(image_dir, nframe, save_dir, save_index_path=None, save_scale_path=None):
    """
    1. 读取所有jpg
    2. 采样nframe帧
    3. 每帧独立计算resize scale
    4. resize + 保存
    5. 返回 content + inputs
    """

    # 1️⃣ 读取图片列表
    image_list = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith(".jpg")
    ])

    # 2️⃣ 采样
    sampled, indices = sample_frames(image_list, nframe, save_index_path)

    os.makedirs(save_dir, exist_ok=True)

    content = []
    inputs = []
    resized_images = []

    scale, _ = compute_resize_scale(sampled[indices[0]])
    print(f"scale: {scale}")

    if save_scale_path is not None:
        os.makedirs(os.path.dirname(save_scale_path), exist_ok=True)  # ✅ 加这一行

        with open(save_scale_path, "w") as f:
            f.write(f"{scale}\n")

    # 3️⃣ 逐帧处理
    for i, img_path in enumerate(sampled):
        # 👉 resize
        img = Image.open(img_path).convert("RGB")
        img = resize_with_scale(img, scale)
        resized_images.append(img)

        # 👉 保存
        save_path = os.path.join(save_dir, f"frame_{i:04d}.jpg")
        img.save(save_path)

        # 👉 组织输入结构
        content.append({
            "type": "image",
            "value": save_path
        })
        inputs.append(save_path)

    return {
        "content": content,
        "inputs": inputs,
        "indices": indices,
        "num_frames": len(sampled),
        "resized_images": resized_images,
        "scale": scale,
    }


# result = process_sequence(
#     image_dir="/nfs/data3/shuaicong/TempRMOT/refer-mot20/MOT20/valid/MOT20-03",
#     nframe=128,
#     save_dir="/nfs/data3/shuaicong/GPT5/test1",
#     save_index_path="/nfs/data3/shuaicong/GPT5/test1/index.txt"
# )
#
# content = result["content"]
# inputs = result["inputs"]
# print(content)

import os

def batch_process_sequences(
    root_dir,
    nframe,
    save_root,
    dataset_name="MOT20"
):
    """
    root_dir: .../MOT20/valid
    save_root: /nfs/data3/shuaicong/GPT5/sampled_images
    """

    sequences = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    all_results = {}

    for seq in sequences:
        image_dir = os.path.join(root_dir, seq)

        save_dir = os.path.join(save_root, dataset_name, seq, f'nframe_{nframe}')
        save_index_path = os.path.join(save_dir, "index.txt")
        save_scale_path = os.path.join(save_dir, "scale.txt")

        os.makedirs(save_dir, exist_ok=True)

        print(f"[Processing] {seq}")

        result = process_sequence(
            image_dir=image_dir,
            nframe=nframe,
            save_dir=save_dir,
            save_index_path=save_index_path,
            save_scale_path=save_scale_path
        )

        all_results[seq] = result

    return all_results

results = batch_process_sequences(
    root_dir="/nfs/data3/shuaicong/TempRMOT/refer-mot20/MOT20/valid",
    nframe=256,
    save_root="/nfs/data3/shuaicong/GPT5/sampled_images",
    dataset_name="MOT20"
)

batch_process_sequences(
    root_dir="/nfs/data3/shuaicong/TempRMOT/refer-ovis/OVIS/valid",
    nframe=256,
    save_root="/nfs/data3/shuaicong/GPT5/sampled_images",
    dataset_name="OVIS"
)

batch_process_sequences(
    root_dir="/nfs/data3/shuaicong/TempRMOT/refer-mot17/MOT17/valid",
    nframe=256,
    save_root="/nfs/data3/shuaicong/GPT5/sampled_images",
    dataset_name="MOT17"
)
