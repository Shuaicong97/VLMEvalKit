# Instructions

## Datasets
1. Download the official OVIS images from [official valid.zip](https://drive.google.com/drive/folders/1eE4lLKCbv54E866XBVce_ebh3oXYq99b?usp=drive_link).
2. Download the official MOT17 images from [official website test folder](https://motchallenge.net/data/MOT17/).
3. Download the official MOT20 images from [official website training folder MOT20-03 and MOT20-05](https://motchallenge.net/data/MOT20/).

👆 Needed videos are listed in data/queries_x.json.

To run the spatial- and temporal grounding tasks using APIs, e.g., Gemini, Claude, GPT-4v, GPT-4o.
Set API keys in [.env](../.env) file. The model names are defined in [config.py](../vlmeval/config.py).
Search for the suitable ones.

Use file [llm_temporal.py](llm_temporal.py), modify the model name:
`model = supported_VLM['Idefics3-8B-Llama3']()`.

We have three datasets: OVIS, MOT17, MOT20. And we only use the **valid** subset.
num_workers is 4 (GPUs) by default.

This repo uses only 16/64 frames to do their tasks, e.g., VQA, MCQ.
But it's not enough for our task.

So please try the following commands. Do the temporal grounding first:

1. Use all images to see if OOM occurs.
`python llm_temporal.py --query_json PATH_TO_QUERIES --image_root PATH_TO_IMAGE --output_dir PATH_TO_OUTPUTS`
2. Use --nframe 256. (try also 128/64/32 as well). A possible summary table looks like Table 5 in [LongVideoBench](https://arxiv.org/pdf/2407.15754).
`python llm_temporal.py --query_json PATH_TO_QUERIES --image_root PATH_TO_IMAGE --nframe 64 --output_dir PATH_TO_OUTPUTS`
3. Use --resize to resize the images and also take all images. (for comparison, the scale should be the same)
`python llm_temporal.py --query_json PATH_TO_QUERIES --image_root PATH_TO_IMAGE --resize --output_dir PATH_TO_OUTPUTS`
4. Use --resize and --nframe. (mostly no need)
`python llm_temporal.py --query_json PATH_TO_QUERIES --image_root PATH_TO_IMAGE --resize --nframe 64 --output_dir PATH_TO_OUTPUTS`

Do the temporal task and use OVIS first.
Then add `--spatial` to do the spatial grounding.
Then test on MOT17 and MOT20.
