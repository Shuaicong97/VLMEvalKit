import os
import numpy as np
from PIL import Image

def sample_frames(image_list, nframe, save_index_path=None):
    total = len(image_list)
    if nframe == -1 or nframe >= total:
        return image_list

    indices = np.linspace(0, total - 1, nframe, dtype=int)
    sampled = [image_list[i] for i in indices]

    if save_index_path is not None:
        with open(save_index_path, "w") as f:
            for idx in indices:
                f.write(f"{idx}\n")
    return sampled, indices.tolist()

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


def process_and_save(sampled_paths, scale_w, scale_h, tmp_dir):
    resized_images = []
    content = []
    inputs = []

    os.makedirs(tmp_dir, exist_ok=True)

    for i, p in enumerate(sampled_paths):
        img = Image.open(p).convert("RGB")

        # resize函数（你需要自己实现或替换）
        img = resize_with_scale(img, scale_w, scale_h)

        resized_images.append(img)

        tmp_path = os.path.join(tmp_dir, f"frame_{i:04d}.jpg")
        img.save(tmp_path)

        content.append({
            "type": "image",
            "value": tmp_path
        })
        inputs.append(tmp_path)

    return resized_images, content, inputs


image_dir = "/nfs/data3/shuaicong/TempRMOT/refer-mot20/MOT20/valid/MOT20-03"
image_list = sorted([
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.endswith(".jpg")
])

sampled, indices = sample_frames(image_list, nframe=128)

resized_images, content, inputs = process_and_save(
    sampled,
    scale_w=640,
    scale_h=360,
    tmp_dir="/nfs/data3/shuaicong/GPT5/test"
)