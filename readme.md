# Instructions

## Datasets
1. Download the official OVIS images from [official valid.zip](https://drive.google.com/drive/folders/1eE4lLKCbv54E866XBVce_ebh3oXYq99b?usp=drive_link).
2. Download the official MOT17 images from [official website test folder](https://motchallenge.net/data/MOT17/).
3. Download the official MOT20 images from [official website training folder MOT20-03 and MOT20-05](https://motchallenge.net/data/MOT20/).

👆 Needed videos are listed in data/queries_x.json.

To run the spatial- and temporal grounding tasks using APIs, e.g., Gemini, Claude, GPT-4v.
Set API keys in [.env](.env) file.

Use file [llm_temporal.py](llm_temporal.py), modify model name in Line 60:
`model = supported_VLM['Idefics3-8B-Llama3']()`.

We have three datasets: OVIS, MOT17, MOT20. And we only use the **valid** subset.
num_workers is 4 (GPUs) by default.

This repo uses only 16/64 frames to do their tasks, e.g., VQA, MCQ.
But it's not enough for our task.

So please try the following commands. Do the temporal grounding first:

1. Use all images to see if OOM occurs.
`python llm_temporal.py --query_json PATH_TO_QUERIES --image_root PATH_TO_IMAGE --output_dir PATH_TO_OUTPUTS`
2. Use --resize to resize the images and also take all images.
`python llm_temporal.py --query_json PATH_TO_QUERIES --image_root PATH_TO_IMAGE --resize --output_dir PATH_TO_OUTPUTS`
3. Use --nframe 64.
`python llm_temporal.py --query_json PATH_TO_QUERIES --image_root PATH_TO_IMAGE --nframe 64 --output_dir PATH_TO_OUTPUTS`
4. Use --resize and --nframe.
`python llm_temporal.py --query_json PATH_TO_QUERIES --image_root PATH_TO_IMAGE --resize --nframe 64 --output_dir PATH_TO_OUTPUTS`

Do the temporal task and use OVIS first.
