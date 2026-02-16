import os
import argparse
from pickle import load, dump
import numpy as np
from data_legacy import binarize
from simulation_utils import PEC_pairs, direct, multipec

def compute_adaptive_seed_pairs(sorted_pairs, min_seeds=10, start_percent=1.0):
    """
    Select the best-ranked pairs instead of thresholding PEC values.
    This avoids the flat-distribution percentile problem.
    """
    pair_items = list(sorted_pairs.items())  # already sorted ascending by PEC
    N = len(pair_items)
    perc = start_percent

    while True:
        k = max(min_seeds, int(N * perc / 100.0))
        seeds = dict(pair_items[:k])

        if len(seeds) >= min_seeds or perc >= 10:
            break
        perc += 0.5

    print(f"Seed percentile used: {perc:.2f}%  -> {len(seeds)} seed pairs (from {N} total)")
    return seeds



def run_multipec_analysis(file, path_load_data, path_save_data):
    """
    Process one EEG or CNN file with MultiPEC analysis.
    """

    # Determine file type
    filename_no_ext = file.replace(".npy", "")
    parts = filename_no_ext.split("_")
    if "eeg" in path_load_data.lower():
        stimulus_type_str, sub = parts[0], parts[1]
        filename_ref = f"{stimulus_type_str}_{sub}"
    elif "cnn" in path_load_data.lower():
        filename_ref = "cnn"
    else:
        raise ValueError("Unknown data source. 'cnn' or 'eeg' should be part of the input path.")

    print(f"\nProcessing {file}")
    epochs = np.load(os.path.join(path_load_data, file))  # shape: (n_epochs, n_channels, n_samples)

    for epoch_idx in range(epochs.shape[0]):
        epoch_data = epochs[epoch_idx]
        print(f"  Epoch {epoch_idx + 1}/{epochs.shape[0]}")
        
        epoch_path = os.path.join(path_save_data, f"{label}_{filename_ref}_epoch{epoch_idx}.p")
        if os.path.isfile(epoch_path):
            print(f"Already processed, skipping.")
            continue

        # Binarize all channels
        signal_bin = {ch: binarize(epoch_data[ch, :]) for ch in range(epoch_data.shape[0])}
        
        # Load or compute PEC pairs (with caching)
        pairs_path = os.path.join(path_save_data, f"pairs_{filename_ref}_epoch{epoch_idx}.p")
        if os.path.isfile(pairs_path):
            with open(pairs_path, "rb") as f: pairs = load(f)
        else:
            pairs = PEC_pairs(signal_bin)
            with open(pairs_path, "wb") as f: dump(pairs, f)

        # Direct pairs and sort them by error
        nodes = list(signal_bin.keys())
        directed_pairs = direct(pairs, nodes)
        sorted_pairs = dict(sorted(directed_pairs.items(), key=lambda item: item[1]))

        # Compute median and std for PEC thresholds
        errors = np.array(list(sorted_pairs.values()))

        if "eeg" in path_load_data.lower():
            seed_pairs = compute_adaptive_seed_pairs(sorted_pairs)
            pair_sets = {
                "nets_adaptive": seed_pairs
            }
        else:
            median_err = np.median(errors)
            std_err = np.std(errors)
            pair_sets = {
                "nets_4down": {s: v for s, v in sorted_pairs.items() if v < median_err - 4 * std_err},
                "nets_34": {s: v for s, v in sorted_pairs.items() if median_err - 4 * std_err < v < median_err - 3 * std_err},
                "nets_23": {s: v for s, v in sorted_pairs.items() if median_err - 3 * std_err < v < median_err - 2 * std_err},
                "nets_12": {s: v for s, v in sorted_pairs.items() if median_err - 2 * std_err < v < median_err - 1 * std_err},
                "nets_05median": {s: v for s, v in sorted_pairs.items() if median_err - 0.05 * std_err < v < median_err + 0.05 * std_err},
                "nets_05up": {s: v for s, v in sorted_pairs.items() if v > median_err + 0.5 * std_err}
            }

        # Shared setup
        labels = list(signal_bin.keys())
        node_signal_bin = signal_bin

        for label, take_pairs in pair_sets.items():
            if not take_pairs:
                print(f"    {label}: No pairs selected, skipping.")
                continue

            nonselected = labels
            nets = multipec(node_signal_bin, take_pairs, nonselected, n_nets=len(take_pairs))
            with open(epoch_path, "wb") as f: dump(nets, f)

if __name__ == "__main__":
    # Input for EEG: data/preprocessed/eeg/
    # Output for EEG: data/output/eeg/
    # Input for CNN: data/preprocessed/cnn/
    # Output for CNN: data/output/cnn/
    # Example: python multipec_nets_epochs.py --load_path data/preprocessed/eeg/ --save_path data/output/eeg/ --subjectects 01 02 03 --stimulusuli S1 S2

    parser = argparse.ArgumentParser(description="Run MultiPEC")
    parser.add_argument("--load_path", required=True, help="Path to directory containing preprocessed files")
    parser.add_argument("--save_path", required=True, help="Path to save analysis results")
    parser.add_argument("--subjects", nargs="*", help="subject IDs to process (e.g. 01 02 10)")
    parser.add_argument("--stimuli", nargs="*", help="stimulus types to process (e.g. S1 S3 S7)")
    
    args = parser.parse_args()
    # List files
    all_files = [f for f in os.listdir(args.load_path) if f.endswith("_preprocessed.npy")]

    # Filter files safely
    file_list = []
    for f in all_files:
        parts = f.replace("_preprocessed.npy", "").split("_")
        if len(parts) < 2:
            continue
        stimulus, subject = parts[0], parts[1]
        if args.subjects and subject not in args.subjects:
            continue
        if args.stimuli and stimulus not in args.stimuli:
            continue
        file_list.append(f)

    # Process each file
    for f in file_list:
        try:
            run_multipec_analysis(f, args.load_path, args.save_path)
        except Exception as e:
            print(f"Error processing {f}: {e}")
