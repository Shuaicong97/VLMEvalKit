from PIL import Image
import argparse
import json
import os
import multiprocessing as mp
import time
from glob import glob
import logging
import numpy as np

from svag.test import sub_lists


def build_spatial_grounding_prompt(query):
    return f"""
Given the query {query}, for each frame, detect and localize all the visual contents described by the given textual query in JSON format. 
If the visual content does not exist in a frame, skip that frame. 
Output Format: 
[
{{
    "object_id": 1, 
    "frames": [
    {{"time": 1, "bbox_2d": [x_min, y_min, x_max, y_max]}},
    {{"time": 2, "bbox_2d": [x_min, y_min, x_max, y_max]}},
    ]
 }},
 {{
    "object_id": 2, 
    "frames": [
    {{"time": 2, "bbox_2d": [x_min, y_min, x_max, y_max]}},
    {{"time": 4, "bbox_2d": [x_min, y_min, x_max, y_max]}},
    ]
 }}]

Notes:
- Do NOT include explanations.
- Only output the JSON object described above.
"""

def build_temporal_grounding_prompt(query):
    return f"""
You are performing temporal grounding for a video.

Event: "{query}"

There may be multiple objects that satisfy the query.

For each object:
- Identify its object_id
- Identify its start frame
- Identify its end frame

Please answer using EXACTLY the following format:
object_id: <id>
start_frame: <time>
end_frame: <time>

One block per object. No explanation.
"""

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

# 1. Gemini
# model = supported_VLM['GeminiPro2-5']()
# 2. GPT-4v
# model = supported_VLM['GPT4V_HIGH']()
# 3. GPT-4o
# model = supported_VLM['ChatGPT4o']()
# 4. Claude
# model = supported_VLM['Claude4_Sonnet']()
from vlmeval.config import supported_VLM
model = supported_VLM['Idefics3-8B-Llama3']()

def load_checkpoint(path):
    finished = set()
    if not os.path.exists(path):
        return finished

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ": " not in line:
                continue
            video_name, query = line.split(": ", 1)
            finished.add((video_name, query))
    return finished

def append_checkpoint(path, video_name, query):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{video_name}: {query}\n")

def sample_frames(image_list, nframe):
    total = len(image_list)
    if nframe == -1 or nframe >= total:
        return image_list

    indices = np.linspace(0, total - 1, nframe, dtype=int)
    sampled = [image_list[i] for i in indices]
    return sampled

def worker_process(worker_id, video_list, args):
    logging.info(f"[Worker {worker_id}] Start processing {len(video_list)} videos")

    checkpoint_path = os.path.join(
        args.output_dir, f"checkpoint_worker_{worker_id}.txt"
    )

    finished_pairs = load_checkpoint(checkpoint_path)

    for video_name, queries in video_list:
        video_id = video_name.split("_")[0]
        image_folder = os.path.join(args.image_root, video_id)

        if not os.path.isdir(image_folder):
            logging.warning(f"[Warning] Image folder not found: {image_folder}")
            continue

        image_paths = sorted(glob(os.path.join(image_folder, "*.jpg")))
        if len(image_paths) == 0:
            logging.warning(f"[Warning] No images found in {image_folder}")
            continue

        logging.info(f"Processing video {video_id}, {len(image_paths)} frames")

        video_results = []

        if args.resize:
            scale_w, scale_h = compute_resize_scale(image_paths[0])
        else:
            scale_w, scale_h = 1.0, 1.0

        resized_images = []
        for p in image_paths:
            img = Image.open(p)
            if args.resize:
                img = resize_with_scale(img, scale_w)
            resized_images.append(img)
        scale = (scale_w, scale_h)

        resized_images = sample_frames(resized_images, args.nframe)
        logging.info(f"[Worker {worker_id}] Video {video_name} uses {args.nframe} frames")

        for query in queries:
            if (video_name, query) in finished_pairs:
                logging.info(
                    f"[Worker {worker_id}] Skip (ckpt): {video_name} | {query}"
                )
                continue

            if args.spatial:
                prompt = build_spatial_grounding_prompt(query)
            else:
                prompt = build_temporal_grounding_prompt(query)

            inputs = resized_images + [prompt]
            ret = model.generate(inputs)

            video_results.append({
                "scale": scale,
                "query": query,
                "response": ret
            })
            append_checkpoint(checkpoint_path, video_name, query)
            finished_pairs.add((video_name, query))
            logging.info(f"[Worker {worker_id}] Video {video_id}, Query: {query}, Response: {ret}")

        output_json_path = os.path.join(args.output_dir, f"{video_name}.json")
        with open(output_json_path, "w") as f:
            json.dump(video_results, f, ensure_ascii=False)

    logging.info(f"[Worker {worker_id}] Done")

def split_videos(video_items, num_workers):
    num_videos = len(video_items)
    per_worker = num_videos // num_workers
    remainder = num_videos % num_workers

    sub_lists = []
    start = 0
    for i in range(num_workers):
        end = start + per_worker + (1 if i < remainder else 0)
        sub_lists.append(video_items[start:end])
        start = end
    return sub_lists

def main():
    parser = argparse.ArgumentParser(description="Video Temporal Grounding with api")
    parser.add_argument("--query_json", type=str, default="data/queries_ovis.json", help="ovis / mot17 / mot20")
    parser.add_argument("--image_root", type=str, default="datasets/OVIS/valid", help="ovis / mot17 / mot20")
    parser.add_argument("--resize", action="store_true", help="enable resize of images according to long_max/short_min rules")
    parser.add_argument("--nframe", type=int, default=-1, help="-1 means all frames, or 16, 64 etc.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--spatial", action="store_true", help="use the spatial grounding prompt")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # logging
    log_filename = os.path.join(
        args.output_dir,
        f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )

    with open(args.query_json, "r", encoding="utf-8") as f:
        queries_data = json.load(f)

    video_items = list(queries_data.items())
    num_workers = args.num_workers

    sub_lists = split_videos(video_items, num_workers)
    processes = []

    logging.info("Start")
    start_time = time.time()
    for i, sub_list in enumerate(sub_lists):
        print(f"Worker {i}: {sub_list}")
        p = mp.Process(target=worker_process, args=(i, sub_list, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    logging.info(f"Over. Time elapsed: {end_time - start_time:.1f}s")



if __name__ == "__main__":
    main()