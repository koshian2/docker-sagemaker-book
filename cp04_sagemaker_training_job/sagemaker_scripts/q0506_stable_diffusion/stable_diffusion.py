# https://github.com/bes-dev/stable_diffusion.openvino/blob/master/demo.py を改変

# -- coding: utf-8 --`
import argparse
import os
# engine
from stable_diffusion_engine import StableDiffusionEngine
# scheduler
from diffusers import LMSDiscreteScheduler, PNDMScheduler
# utils
import cv2
import os
import json
import numpy as np


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
    if args.init_image is None:
        scheduler = LMSDiscreteScheduler(
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            tensor_format="np"
        )
    else:
        scheduler = PNDMScheduler(
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            skip_prk_steps = True,
            tensor_format="np"
        )
    engine = StableDiffusionEngine(
        model = args.model,
        scheduler = scheduler,
        tokenizer = args.tokenizer
    )

    for i in range(args.n_out_imgs):
        print("\n--- image", i+1, "/", args.n_out_imgs, "---")
        image = engine(
            prompt = args.prompt,
            init_image = None if args.init_image is None else cv2.imread(args.init_image),
            mask = None if args.mask is None else cv2.imread(args.mask, 0),
            strength = args.strength,
            num_inference_steps = args.num_inference_steps,
            guidance_scale = args.guidance_scale,
            eta = args.eta
        )
        if args.n_out_imgs > 1:
            splits = os.path.splitext(args.output)
            outpath = f"{splits[0]}_{i:04}{splits[1]}"
        else:
            outpath = args.output
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        cv2.imwrite(outpath, image)

    # Write prompts
    with open(os.path.join(os.path.dirname(args.output), "arguments.json"), "w", encoding="utf-8") as fp:
        json.dump(vars(args), fp, ensure_ascii=False, indent=4, separators=(',', ': '))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # pipeline configure
    parser.add_argument("--model", type=str, default="bes-dev/stable-diffusion-v1-4-openvino", help="model name")
    # randomizer params
    parser.add_argument("--seed", type=int, default=None, help="random seed for generating consistent images per prompt")
    # scheduler params
    parser.add_argument("--beta-start", type=float, default=0.00085, help="LMSDiscreteScheduler::beta_start")
    parser.add_argument("--beta-end", type=float, default=0.012, help="LMSDiscreteScheduler::beta_end")
    parser.add_argument("--beta-schedule", type=str, default="scaled_linear", help="LMSDiscreteScheduler::beta_schedule")
    # diffusion params
    parser.add_argument("--num-inference-steps", type=int, default=32, help="num inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="guidance scale")
    parser.add_argument("--eta", type=float, default=0.0, help="eta")
    # tokenizer
    parser.add_argument("--tokenizer", type=str, default="openai/clip-vit-large-patch14", help="tokenizer")
    # prompt
    parser.add_argument("--prompt", type=str, default="Street-art painting of Emilia Clarke in style of Banksy, photorealism", help="prompt")
    # img2img params
    parser.add_argument("--init-image", type=str, default=None, help="path to initial image")
    parser.add_argument("--strength", type=float, default=0.5, help="how strong the initial image should be noised [0.0, 1.0]")
    # inpainting
    parser.add_argument("--mask", type=str, default=None, help="mask of the region to inpaint on the initial image")
    # output name
    parser.add_argument("--output", type=str, default="output.png", help="output image name")
    parser.add_argument("--n_out_imgs", type=int, default=1, help="number of output images")
    args = parser.parse_args()

    if "SM_HPS" in os.environ.keys():
        hps = json.loads(os.environ["SM_HPS"])
        for key, value in hps.items():
            if args.__contains__(key):
                args.__setattr__(key, value)    

    print(args)

    main(args)