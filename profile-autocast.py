from typing import List
from bioclip import TreeOfLifeClassifier, Rank
import glob
import argparse
import os
import torch
import time
import json
import pandas as pd
import datetime


PROFILE_CONFIG_PATH = "profile.json"


class Config(object):
    def __init__(self, config_dict):
        self.num_images = config_dict.get("num_images", 100)
        self.device = config_dict.get("device", "cpu")
        self.autocast = config_dict.get("autocast", False)
        self.batch_size = config_dict.get("batch_size", 10)


def read_profile_configs() -> List[Config]:
    configs = []
    if os.path.exists(PROFILE_CONFIG_PATH):
        with open(PROFILE_CONFIG_PATH, 'r') as f:
            data = json.load(f)
            for config_dict in data:
                config = Config(config_dict)
                configs.append(config)
        return configs
    else:
        return {}


def profile_prediction(device, use_autocast, batch_size, image_paths, attempt):
    classifier = TreeOfLifeClassifier(device=device, enable_autocast=use_autocast)
    
    #if use_autocast:
    #    classifier.txt_embeddings = classifier.txt_embeddings.to(torch.get_autocast_dtype(device))


    start_time = time.time()

    classifier.predict(image_paths, Rank.SPECIES, k=1, batch_size=batch_size)
    end_time = time.time()

    elapsed = end_time - start_time
    images_per_second = len(image_paths) / elapsed
    return {
        "device": device,
        "use_autocast": use_autocast,
        "autocast_dtype": torch.get_autocast_dtype(device) if use_autocast else None,
        "batch_size": batch_size,
        "num_images": len(image_paths),
        "elapsed_time": elapsed,
        "attempt": attempt,
        "images_per_second": images_per_second
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process images with TreeOfLifeClassifier')
    parser.add_argument('--dir', type=str, required=True,
                        help='Directory containing images to process')
    return parser.parse_args()


def main(): 
    args = parse_arguments()
    image_paths = glob.glob(os.path.join(args.dir, "**/*.jpg"), recursive=True)
    results = []
    with open(PROFILE_CONFIG_PATH, 'r') as f:    
        data = json.load(f)
        for num_images in data.get("num_images", 100):
            for autocast in data.get("autocast", False):
                for device in data.get("device", "cpu"):
                    for batch_size in data.get("batch_size", 10):
                        for attempt in range(data.get("attempts", 1)):
                            print(f"Processing {num_images} images with device {device}, autocast {autocast}, batch size {batch_size}, attempt {attempt}")
                            # Call the function with the current configuration
                            result = profile_prediction(
                                device=device,
                                use_autocast=autocast,
                                batch_size=batch_size,
                                image_paths=image_paths[:num_images],
                                attempt=attempt
                            )
                            results.append(result)
    df = pd.DataFrame(results)
    print("Results")
    print(df)
    results_filename = f"profile_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(results_filename, index=False)
    print(f"Results saved to {results_filename}")


if __name__ == "__main__":
    main()