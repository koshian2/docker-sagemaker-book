import argparse
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Some Training")
    parser.add_argument("--data_augmentation_type", type=str, default="default_value")
    parser.add_argument("--backbone", type=str, default="default_value")
    parser.add_argument("--batch_size", type=int, default=8)

    opt = parser.parse_args()

    if "SM_HPS" in os.environ.keys():
        hps = json.loads(os.environ["SM_HPS"])
        for key, value in hps.items():
            if opt.__contains__(key):
                opt.__setattr__(key, value)

    print(opt)
