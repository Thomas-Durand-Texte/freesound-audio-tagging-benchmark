"""Exploration and analysis module for dataset statistics and visualizations."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.core.utils import load_config
from src.data import dataset as data
from src.data.dataset import DatasetType
from src.features import signal_tools, spectrogram_optimized


def compute_dataset_statistics(config: dict, dataset_types: list[DatasetType]) -> dict:
    """Compute comprehensive statistics for specified datasets.

    Args:
        config: Configuration dictionary
        dataset_types: List of dataset types to analyze ('train_curated', 'train_noisy')

    Returns:
        Dictionary containing statistics for each dataset
    """
    print("\n" + "=" * 70)
    print("Computing Dataset Statistics")
    print("=" * 70)

    dataset_config = data.AudioDatasetConfig.from_dict(config["data"])
    metadata_manager = data.MetadataManager(dataset_config)

    # Get problematic files info
    problematic_df = metadata_manager.get_problematic_files()
    problematic_set = set(problematic_df["filename"].tolist()) if len(problematic_df) > 0 else set()

    all_stats = {}

    for dataset_type in dataset_types:
        print(f"\n{'-' * 70}")
        print(f"Processing: {dataset_type}")
        print(f"{'-' * 70}")

        dataset = data.AudioDataset(dataset_config, dataset_type=dataset_type)

        # Get label statistics using SQL
        label_stats_df = metadata_manager.get_label_statistics(dataset_type, skip_problematic=False)
        label_counts = dict(zip(label_stats_df["label"], label_stats_df["count"]))

        # Initialize statistics collectors
        sample_rates = []
        durations = []
        file_info = []
        problematic_found = []

        # Collect data for each file
        print(f"Analyzing {len(dataset)} files...")
        for idx in tqdm(range(len(dataset)), desc=f"{dataset_type}"):
            row = dataset.metadata.iloc[idx]
            filename = row["fname"]
            labels = row["labels"].split(",")

            # Check if file is problematic
            is_problematic = filename in problematic_set
            problem_type = None
            if is_problematic:
                prob_info = problematic_df[problematic_df["filename"] == filename].iloc[0]
                problem_type = prob_info["problem_type"]
                problematic_found.append(
                    {
                        "filename": filename,
                        "type": problem_type,
                        "labels": labels,
                    }
                )

            # Get audio info (without loading full waveform)
            sr, duration = dataset.audio_loader.get_audio_info(filename)

            sample_rates.append(sr)
            durations.append(duration)

            file_info.append(
                {
                    "filename": filename,
                    "labels": labels,
                    "sample_rate": sr,
                    "duration": duration,
                    "is_problematic": is_problematic,
                    "problem_type": problem_type,
                }
            )

        # Convert to numpy arrays for statistics
        sample_rates = np.array(sample_rates)
        durations = np.array(durations)

        # Compute statistics
        stats = {
            "dataset_type": dataset_type,
            "total_files": len(dataset),
            "unique_labels": len(label_counts),
            "label_counts": label_counts,
            "sample_rates": {
                "unique": sorted(np.unique(sample_rates).tolist()),
                "mean": float(np.mean(sample_rates)),
                "median": float(np.median(sample_rates)),
                "std": float(np.std(sample_rates)),
            },
            "durations": {
                "sum_hours": float(np.sum(durations) / 3600),
                "mean": float(np.mean(durations)),
                "median": float(np.median(durations)),
                "std": float(np.std(durations)),
                "min": float(np.min(durations)),
                "max": float(np.max(durations)),
                "percentile_25": float(np.percentile(durations, 25)),
                "percentile_75": float(np.percentile(durations, 75)),
                "percentile_95": float(np.percentile(durations, 95)),
            },
            "problematic_files": problematic_found,
            "file_info": file_info,
        }

        all_stats[dataset_type] = stats

        # Compute sample rate distribution for saved statistics
        sr_distribution = {}
        for sr in np.unique(sample_rates):
            sr_distribution[int(sr)] = int(np.sum(sample_rates == sr))
        stats["sample_rates"]["distribution"] = sr_distribution

        # Print summary
        print(f"\nSummary for {dataset_type}:")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Unique labels: {stats['unique_labels']}")
        print(f"  Total duration: {stats['durations']['sum_hours']:.2f} hours")
        print(
            f"  Duration range: {stats['durations']['min']:.2f}s - {stats['durations']['max']:.2f}s"
        )
        print(f"  Sample rates: {stats['sample_rates']['unique']}")

        # Warn about problematic files
        if problematic_found:
            print(f"\n  ⚠️  WARNING: Found {len(problematic_found)} problematic files:")
            for prob in problematic_found:
                print(
                    f"    - {prob['filename']}: {prob['type']} (labels: {', '.join(prob['labels'])})"
                )

    return all_stats


def save_statistics(stats: dict, output_path: Path) -> None:
    """Save statistics to JSON file.

    Args:
        stats: Statistics dictionary
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {output_path}")


def load_statistics(stats_path: Path) -> dict:
    """Load statistics from JSON file.

    Args:
        stats_path: Path to JSON file

    Returns:
        Statistics dictionary
    """
    with open(stats_path) as f:
        stats = json.load(f)
    print(f"\nStatistics loaded from: {stats_path}")
    return stats


def display_statistics(stats: dict) -> None:
    """Display comprehensive statistics with plots.

    Args:
        stats: Statistics dictionary
    """
    print("\n" + "=" * 70)
    print("Dataset Statistics Summary")
    print("=" * 70)

    for dataset_type, dataset_stats in stats.items():
        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset_type}")
        print(f"{'=' * 70}")

        print(f"\n  Total files: {dataset_stats['total_files']}")
        print(f"  Unique labels: {dataset_stats['unique_labels']}")

        # Duration statistics
        dur = dataset_stats["durations"]
        print("\n  Duration Statistics:")
        print(f"    Total: {dur['sum_hours']:.2f} hours")
        print(f"    Mean: {dur['mean']:.3f}s")
        print(f"    Median: {dur['median']:.3f}s")
        print(f"    Std: {dur['std']:.3f}s")
        print(f"    Range: [{dur['min']:.3f}s, {dur['max']:.3f}s]")
        print(f"    25th percentile: {dur['percentile_25']:.3f}s")
        print(f"    75th percentile: {dur['percentile_75']:.3f}s")
        print(f"    95th percentile: {dur['percentile_95']:.3f}s")

        # Sample rate statistics
        sr = dataset_stats["sample_rates"]
        print("\n  Sample Rate Statistics:")
        print(f"    Unique values: {sr['unique']}")
        print(f"    Mean: {sr['mean']:.1f} Hz")
        print(f"    Median: {sr['median']:.1f} Hz")

        # Check sample rate consistency
        unique_sr = sr["unique"]
        if len(unique_sr) == 1:
            print(f"    ✓ All files have the same sample rate: {unique_sr[0]} Hz")
        else:
            print(f"    ⚠️  WARNING: Files have different sample rates: {unique_sr}")
            if "distribution" in sr:
                print(f"       Distribution: {sr['distribution']}")

        # Top 20 labels
        print("\n  Top 20 Labels by Count:")
        label_counts = dataset_stats["label_counts"]
        for i, (label, count) in enumerate(list(label_counts.items())[:20], 1):
            print(f"    {i:2d}. {label:30s}: {count:4d} files")

    # Create visualizations
    print(f"\n{'-' * 70}")
    print("Creating visualizations...")

    for dataset_type, dataset_stats in stats.items():
        # Figure 1: Duration distribution
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        durations = [info["duration"] for info in dataset_stats["file_info"]]

        ax1.hist(durations, bins=50, edgecolor="black", alpha=0.7)
        ax1.set_xlabel("Duration (seconds)")
        ax1.set_ylabel("Count")
        ax1.set_title(f"Duration Distribution - {dataset_type}")
        ax1.grid(True, alpha=0.3)

        ax2.boxplot(durations, vert=True)
        ax2.set_ylabel("Duration (seconds)")
        ax2.set_title(f"Duration Boxplot - {dataset_type}")
        ax2.grid(True, alpha=0.3)

        fig1.tight_layout()
        plt.show()

        # Figure 2: Top labels distribution
        fig2, ax = plt.subplots(figsize=(14, 8))

        label_counts = dataset_stats["label_counts"]
        top_n = 30
        top_labels = list(label_counts.items())[:top_n]
        labels, counts = zip(*top_labels)

        y_pos = np.arange(len(labels))
        ax.barh(y_pos, counts, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Number of Files")
        ax.set_title(f"Top {top_n} Labels by Count - {dataset_type}")
        ax.grid(True, alpha=0.3, axis="x")

        fig2.tight_layout()
        plt.show()


def compute_spectrograms(
    config: dict,
    output_dir: Path,
    max_files_per_label: int = 5,
    skip_problematic: bool = True,
) -> None:
    """Generate and save spectrograms for curated dataset.

    Args:
        config: Configuration dictionary
        output_dir: Base directory to save spectrograms
        max_files_per_label: Maximum number of files to process per label
        skip_problematic: Whether to skip known problematic files
    """
    print("\n" + "=" * 70)
    print("Computing Spectrograms for Curated Dataset")
    print("=" * 70)
    print(f"  Skip problematic files: {skip_problematic}")

    # Load dataset
    dataset_config = data.AudioDatasetConfig.from_dict(config["data"])
    dataset = data.AudioDataset(dataset_config, dataset_type="train_curated")

    # Get problematic files from metadata manager
    metadata_manager = data.MetadataManager(dataset_config)
    problematic_df = metadata_manager.get_problematic_files()
    problematic_set = set(problematic_df["filename"].tolist()) if len(problematic_df) > 0 else set()

    # Spectrogram parameters from config
    spectrogram_config = config.get("spectrogram", {})
    f_min = spectrogram_config.get("f_min", 20.0)
    f_max = spectrogram_config.get("f_max", 8000.0)
    f_mid = spectrogram_config.get("f_mid")
    n_bands = spectrogram_config.get("n_bands", 128)
    signal_duration = spectrogram_config.get("signal_duration", 3.0)
    hop_length = spectrogram_config.get("hop_length", 512)

    sample_rate = config["data"]["sample_rate"]

    print("\nSpectrogram parameters:")
    print(f"  Frequency range: {f_min} - {f_max} Hz")
    if f_mid is not None:
        print(f"  Middle frequency: {f_mid} Hz (dual-range mode)")
    print(f"  Number of bands: {n_bands}")
    print(f"  Signal duration: {signal_duration}s")
    print(f"  Hop length: {hop_length}")

    # Initialize multi-resolution filter bank
    print("\nInitializing SuperGaussian filter bank...")
    mr_filter_bank = spectrogram_optimized.MultiResolutionFilterBank(
        envelope_class=signal_tools.SuperGaussianEnvelope,
        f_min=f_min,
        f_max=f_max,
        f_mid=f_mid,
        num_bands=n_bands,
        sample_rate=sample_rate,
        signal_duration=signal_duration,
        spectrum_threshold=0.001,
    )
    print(f"  Downsample levels: {mr_filter_bank.downsample_levels}")

    # Group files by label
    print("\nGrouping files by label...")
    label_files = defaultdict(list)
    for idx in range(len(dataset)):
        row = dataset.metadata.iloc[idx]
        filename = row["fname"]
        labels = row["labels"].split(",")
        for label in labels:
            label_files[label].append((idx, filename))

    print(f"  Found {len(label_files)} unique labels")

    # Count skipped files
    total_skipped = 0

    # Process each label
    for label, files in tqdm(label_files.items(), desc="Processing labels"):
        # Create output directory for this label
        label_dir = output_dir / label.replace("/", "_")
        label_dir.mkdir(parents=True, exist_ok=True)

        # Filter out problematic files if requested
        files_to_process = []
        for idx, filename in files:
            if skip_problematic and filename in problematic_set:
                total_skipped += 1
                continue
            files_to_process.append((idx, filename))

        # Process up to max_files_per_label files
        for idx, filename in files_to_process[:max_files_per_label]:
            try:
                # Get labels for this file
                row = dataset.metadata.iloc[idx]
                file_labels = row["labels"].split(",")
                labels_str = ", ".join(file_labels)

                # Load audio
                waveform, sr = dataset.audio_loader.load_audio(filename)

                # Limit to first N seconds
                max_samples = int(signal_duration * sr)
                if len(waveform) > max_samples:
                    waveform = waveform[:max_samples]

                # Compute spectrogram
                spec, time_step, _ = mr_filter_bank.compute_spectrogram(waveform, hop_length)
                time_end = float(time_step * (spec.shape[1] - 1))

                # Compute FFT for spectrum plot
                waveform_fft = np.fft.rfft(waveform)
                n_fft = len(waveform)
                freqs = np.fft.rfftfreq(n_fft, 1 / sr)

                # Create comprehensive figure
                fig = plt.figure(figsize=(16, 12))
                gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

                # 1. Spectrogram
                ax1 = fig.add_subplot(gs[0, :])
                spec_max = np.max(spec)
                spec_vmin = spec_max - 4.0  # 40 dB range

                img = ax1.imshow(
                    spec,
                    aspect="auto",
                    origin="lower",
                    extent=(0.0, time_end, 0.0, n_bands - 1.0),
                    cmap="viridis",
                    vmin=spec_vmin,
                    vmax=spec_max,
                )

                # Add frequency ticks
                freq_ticks_hz = np.array([20, 50, 100, 200, 500, 1000, 2000, 5000, 8000])
                freq_ticks_hz = freq_ticks_hz[(freq_ticks_hz >= f_min) & (freq_ticks_hz <= f_max)]
                band_indices = [
                    np.argmin(np.abs(mr_filter_bank.center_frequencies - f)) for f in freq_ticks_hz
                ]

                ax1.set_yticks(band_indices)
                ax1.set_yticklabels([f"{int(f)}" for f in freq_ticks_hz])
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Frequency (Hz)")
                ax1.set_title(f"{filename} - Labels: {labels_str}")
                plt.colorbar(img, ax=ax1, label="Magnitude (cB)")

                # 2. Spectrum magnitude (loglog)
                ax2 = fig.add_subplot(gs[1, 0])
                ax2.loglog(freqs[1:], np.abs(waveform_fft[1:]), linewidth=0.5, alpha=0.7)
                ax2.set_xlabel("Frequency (Hz)")
                ax2.set_ylabel("Magnitude")
                ax2.set_title("Spectrum Magnitude (loglog)")
                ax2.grid(True, alpha=0.3, which="both")
                ax2.set_xlim((f_min, f_max))

                # 3. Spectrum real part (semilogx)
                ax3 = fig.add_subplot(gs[1, 1])
                ax3.semilogx(freqs[1:], waveform_fft.real[1:], linewidth=0.5, alpha=0.7)
                ax3.set_xlabel("Frequency (Hz)")
                ax3.set_ylabel("Real Part")
                ax3.set_title("Spectrum Real Part (semilogx)")
                ax3.grid(True, alpha=0.3, which="both")
                ax3.set_xlim((f_min, f_max))

                # 4. Spectrum imaginary part (semilogx)
                ax4 = fig.add_subplot(gs[2, 0])
                ax4.semilogx(freqs[1:], waveform_fft.imag[1:], linewidth=0.5, alpha=0.7)
                ax4.set_xlabel("Frequency (Hz)")
                ax4.set_ylabel("Imaginary Part")
                ax4.set_title("Spectrum Imaginary Part (semilogx)")
                ax4.grid(True, alpha=0.3, which="both")
                ax4.set_xlim((f_min, f_max))

                # 5. Waveform
                ax5 = fig.add_subplot(gs[2, 1])
                time = np.arange(len(waveform)) / sr
                ax5.plot(time, waveform, linewidth=0.5, alpha=0.7)
                ax5.set_xlabel("Time (s)")
                ax5.set_ylabel("Amplitude")
                ax5.set_title("Waveform")
                ax5.grid(True, alpha=0.3)

                # Save figure
                output_path = label_dir / f"{filename.replace('.wav', '.png')}"
                fig.savefig(output_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

            except Exception as e:
                print(f"    Error processing {filename}: {e}")
                continue

    print(f"\nSpectrograms saved to: {output_dir}")
    if skip_problematic and total_skipped > 0:
        print(f"Skipped {total_skipped} problematic file(s)")


def main() -> None:
    """Main exploration entry point."""
    parser = argparse.ArgumentParser(description="Dataset exploration and visualization")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")

    # Statistics options
    parser.add_argument(
        "--compute-stats",
        action="store_true",
        help="Compute dataset statistics",
    )
    parser.add_argument(
        "--display-stats",
        action="store_true",
        help="Display statistics (requires previously computed stats)",
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        default="data/statistics.json",
        help="Path to save/load statistics",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["train_curated", "train_noisy"],
        choices=["train_curated", "train_noisy"],
        help="Datasets to analyze",
    )

    # Spectrogram options
    parser.add_argument(
        "--compute-spectrograms",
        action="store_true",
        help="Compute and save spectrograms for curated dataset",
    )
    parser.add_argument(
        "--spectrogram-dir",
        type=str,
        default="data/spectrograms",
        help="Directory to save spectrograms",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=5,
        help="Maximum files to process per label",
    )
    parser.add_argument(
        "--skip-problematic",
        action="store_true",
        default=True,
        help="Skip known problematic files (default: True)",
    )
    parser.add_argument(
        "--no-skip-problematic",
        dest="skip_problematic",
        action="store_false",
        help="Process all files including known problematic ones",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    # Compute statistics
    if args.compute_stats:
        stats = compute_dataset_statistics(config, args.datasets)
        save_statistics(stats, Path(args.stats_file))

    # Display statistics
    if args.display_stats:
        stats = load_statistics(Path(args.stats_file))
        display_statistics(stats)

    # Compute spectrograms
    if args.compute_spectrograms:
        compute_spectrograms(
            config,
            Path(args.spectrogram_dir),
            max_files_per_label=args.max_per_label,
            skip_problematic=args.skip_problematic,
        )


if __name__ == "__main__":
    main()
