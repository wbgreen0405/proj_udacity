#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import wandb

import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact.
    logger.info("Downloading artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    
    # Load the data
    logger.info("Loading the data")
    df  = pd.read_csv(artifact_local_path, index_col="id")

    # Preprocessing
    logger.info("Start pre-processing")
    # Drop outliers
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Prepare file to be saved
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    filename = args.output_artifact
    df.to_csv(filename)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )

    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)
    artifact.wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")

    # Parameter 1
    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Input dataset",
        required=True
    )

    # Parameter 2
    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Output dataset",
        required=True
    )

    # Parameter 3
    parser.add_argument(
        "--output_type", 
        type=str,
        help="wandb artifact type",
        required=True
    )

    # Parameter 4
    parser.add_argument(
        "--output_description", 
        type=str,
        help="wandb artifact description",
        required=True
    )

    # Parameter 5
    parser.add_argument(
        "--min_price", 
        type=float,
        help="lower limit tolerant for outlier",
        required=True
    )

    # Parameter 6
    parser.add_argument(
        "--max_price", 
        type=float,
        help="superior limit tolerant for outlier",
        required=True
    )

    args = parser.parse_args()

    go(args)
