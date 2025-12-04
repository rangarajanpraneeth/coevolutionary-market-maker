#!/usr/bin/env python3

import argparse
import os
import random
from datetime import datetime

import numpy as np
import tensorflow as tf

from simulation import Market

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run simulation on a specified OHLCV CSV file."
    )
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="Path to OHLCV CSV file (ex: data/NVDA/nvda_up.csv)",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=0, help="Random seed (default: 0)"
    )
    args = parser.parse_args()

    DATA_FILE = args.file
    SEED = args.seed

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"CSV file not found: {DATA_FILE}")

    base = os.path.basename(DATA_FILE)
    scenario = os.path.splitext(base)[0].lower()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join("output", f"{scenario}_{SEED}_{ts}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nUsing dataset: {DATA_FILE}")
    print(f"Scenario name: {scenario}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Seed: {SEED}\n")

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    drl_params = {
        "gamma": 0.95,
        "epsilon_decay": 0.995,
        "learning_rate": 0.0005,
        "replay_buffer_maxlen": 20000,
        "batch_size": 32,
        "epsilon_min": 0.01,
    }

    nn_architecture = [
        {"units": 64, "activation": "relu"},
        {"units": 64, "activation": "relu"},
    ]

    print("Launching simulation...\n")

    try:
        market_sim = Market(
            nn_architecture=nn_architecture,
            drl_params=drl_params,
            ohlcv_filename=DATA_FILE,
            output_dir=OUTPUT_DIR,
        )
        market_sim.run_simulation()
        print(f"\nCompleted — results stored in:\n{OUTPUT_DIR}\n")
    except Exception as e:
        print(f"\nSimulation failed: {e}")
        import traceback

        traceback.print_exc()
