"""Development and testing module for audio dataset functionality."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch

from src.core.config import Config
from src.data.dataset import AudioDataset
from src.features import signal_tools, spectrogram, spectrogram_optimized


def play_audio_file(audio_path: Path) -> str:
    """Play audio file in terminal using system player with interactive controls.

    Args:
        audio_path: Path to audio file

    Returns:
        'continue' to play next audio, 'quit' to stop playback loop, 'error' on failure
    """
    if sys.platform != "darwin":
        print(f"  Audio playback not supported on {sys.platform}")
        print(f"  File path: {audio_path}")
        return "error"

    try:
        import select
        import termios
        import tty

        # Start audio playback process
        process = subprocess.Popen(
            ["afplay", str(audio_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            # Set terminal to raw mode for immediate key detection
            tty.setraw(sys.stdin.fileno())

            print("  [Press 's' to skip, 'q' to quit, or wait for audio to finish]")

            # Poll for process completion and keyboard input
            while process.poll() is None:
                # Check if input is available (non-blocking)
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1).lower()

                    if char == "s":
                        process.terminate()
                        process.wait()
                        print("\r  Skipped                                                  ")
                        return "continue"
                    elif char == "q":
                        process.terminate()
                        process.wait()
                        print("\r  Quit playback                                            ")
                        return "quit"

            # Audio finished naturally
            print("\r  Finished                                                 ")
            return "continue"

        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    except FileNotFoundError:
        print("  Error: afplay not found (macOS only)")
        return "error"
    except Exception as e:
        print(f"  Error during playback: {e}")
        return "error"


def test_audio_loading(config: Config, n_samples: int = 3) -> None:
    """Test audio dataset loading and display info for n samples.

    Args:
        config: Configuration object
        n_samples: Number of samples to test
    """
    print("\n" + "=" * 60)
    print("Testing AudioDataset")
    print("=" * 60)

    # Create AudioDataset instance using config.data directly
    dataset = AudioDataset(config.data, dataset_type="train_curated")

    print(f"\nDataset size: {len(dataset)} samples")
    print(f"Vocabulary size: {len(dataset.metadata_manager.vocabulary)} labels")

    print(f"\nTesting first {n_samples} samples:")
    print("-" * 60)

    for i in range(min(n_samples, len(dataset))):
        row = dataset.metadata.iloc[i]
        filename = row["fname"]
        labels = row["labels"]

        # Get audio info without loading waveform
        sample_rate, duration = dataset.audio_loader.get_audio_info(filename)

        print(f"\nSample {i + 1}:")
        print(f"  Filename: {filename}")
        print(f"  Labels: {labels}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Duration: {duration:.2f} seconds")

        # Load audio waveform
        waveform, _sr = dataset.audio_loader.load_audio(filename)
        print(f"  Waveform shape: {waveform.shape}")
        print(f"  Waveform range: [{waveform.min():.3f}, {waveform.max():.3f}]")

    # Terminal audio playback
    print("\n" + "=" * 60)
    print("Audio playback test")
    print("=" * 60)

    for i in range(min(n_samples, len(dataset))):
        row = dataset.metadata.iloc[i]
        filename = row["fname"]
        labels = row["labels"]
        audio_path = dataset.audio_loader.get_audio_path(filename)

        print(f"\nSample {i + 1}: {filename}")
        print(f"  Labels: {labels}")
        print("  Playing...")

        result = play_audio_file(audio_path)

        if result == "quit":
            print("\nPlayback stopped by user")
            break
        elif result == "error":
            print("\nPlayback error - stopping")
            break
        # result == "continue" -> play next audio


def test_spectrogram_comparison(config: Config, n_samples: int = 3) -> None:
    """Test and compare spectrograms using custom SuperGaussian vs librosa.

    Args:
        config: Configuration object
        n_samples: Number of audio samples to compare
    """
    print("\n" + "=" * 70)
    print("Spectrogram Comparison Test")
    print("=" * 70)

    # Create AudioDataset instance using config.data directly
    dataset = AudioDataset(config.data, dataset_type="train_curated")

    print(f"\nDataset size: {len(dataset)} samples")
    print(f"Testing {n_samples} samples for spectrogram comparison")

    # Spectrogram parameters from config using dot notation
    f_min = config.spectrogram.f_min
    f_max = config.spectrogram.f_max
    f_mid = config.spectrogram.f_mid
    n_bands = config.spectrogram.n_bands
    signal_duration = config.spectrogram.signal_duration
    hop_length = config.spectrogram.hop_length

    print("\nSpectrogram parameters:")
    print(f"  f_min: {f_min} Hz")
    print(f"  f_max: {f_max} Hz")
    print(f"  n_bands: {n_bands}")
    print(f"  signal_duration: {signal_duration} s")
    print(f"  hop_length: {hop_length}")

    for i in range(min(n_samples, len(dataset))):
        row = dataset.metadata.iloc[i]
        filename = row["fname"]
        labels = row["labels"]
        filename = "f2f0a2b1.wav"
        # Load audio waveform
        waveform, sample_rate = dataset.audio_loader.load_audio(filename)

        # Limit to first 5 seconds for faster computation
        max_samples = int(5.0 * sample_rate)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        # Create label for visualization
        audio_label = f"{filename} - {labels}"

        # Compare spectrograms
        spectrogram.compare_spectrograms(
            waveform=waveform,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            f_mid=f_mid,
            n_bands=n_bands,
            audio_label=audio_label,
            signal_duration=signal_duration,
            hop_length=hop_length,
        )


def test_spectrogram_benchmark(
    config: Config, n_samples: int = 1, test_gpu: bool = False, test_multiresolution: bool = True
) -> None:
    """Benchmark optimized spectrogram methods.

    Args:
        config: Configuration object
        n_samples: Number of audio samples to benchmark
        test_gpu: Whether to test GPU acceleration
        test_multiresolution: Whether to test multi-resolution method
    """
    print("\n" + "=" * 70)
    print("Spectrogram Methods Benchmark")
    print("=" * 70)

    # Create AudioDataset instance using config.data directly
    dataset = AudioDataset(config.data, dataset_type="train_curated")

    print(f"\nDataset size: {len(dataset)} samples")
    print(f"Benchmarking {n_samples} sample(s)")

    # Spectrogram parameters from config using dot notation
    f_min = config.spectrogram.f_min
    f_max = config.spectrogram.f_max
    n_bands = config.spectrogram.n_bands
    hop_length = config.spectrogram.hop_length
    n_fft = config.spectrogram.n_fft
    signal_duration = config.spectrogram.signal_duration

    for i in range(min(n_samples, len(dataset))):
        row = dataset.metadata.iloc[i]
        filename = row["fname"]
        labels = row["labels"]

        # Load audio waveform
        waveform, sample_rate = dataset.audio_loader.load_audio(filename)

        # Limit to first 5 seconds for reasonable benchmark time
        max_samples = int(5.0 * sample_rate)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        print(f"\n{'=' * 70}")
        print(f"Sample {i + 1}: {filename}")
        print(f"Labels: {labels}")
        print(f"{'=' * 70}")

        # Initialize filter bank (outside timing)
        print("\nInitializing SuperGaussian filter bank...")
        import time

        init_start = time.perf_counter()
        filter_bank = signal_tools.LogSpacedFilterBank(
            envelope_class=signal_tools.SuperGaussianEnvelope,
            f_min=f_min,
            f_max=f_max,
            num_bands=n_bands,
            sample_rate=sample_rate,
        )
        init_time = time.perf_counter() - init_start
        print(f"Filter bank initialization: {init_time * 1000:.2f} ms (done once)")

        # Run benchmark
        results = spectrogram_optimized.benchmark_spectrogram_methods(
            waveform=waveform,
            filter_bank=filter_bank,
            hop_length=hop_length,
            n_fft=n_fft,
            spectrum_threshold=0.01,
            test_gpu=test_gpu,
            gpu_device="mps",
        )

        # Test multi-resolution method if requested
        if test_multiresolution:
            print(f"\n{'=' * 70}")
            print("Testing Multi-Resolution Method")
            print(f"{'=' * 70}")
            print(f"Signal duration: {signal_duration} seconds")
            print("Initializing multi-resolution filter bank...")

            init_start = time.perf_counter()
            mr_filter_bank = spectrogram_optimized.MultiResolutionFilterBank(
                envelope_class=signal_tools.SuperGaussianEnvelope,
                f_min=f_min,
                f_max=f_max,
                num_bands=n_bands,
                sample_rate=sample_rate,
                signal_duration=signal_duration,
                spectrum_threshold=0.001,
            )
            init_time = time.perf_counter() - init_start
            print(f"  Initialization time: {init_time * 1000:.2f} ms (done once)")
            print(f"  Downsample levels used: {mr_filter_bank.downsample_levels}")

            # Compute spectrogram
            waveform = torch.from_numpy(waveform).to(mr_filter_bank.device)
            # warmup
            spec, time_step, comp_time = mr_filter_bank.compute_spectrogram(waveform, hop_length)
            # compute
            spec, time_step, comp_time = mr_filter_bank.compute_spectrogram(waveform, hop_length)
            print(f"  Computation time: {comp_time * 1000:.2f} ms")
            print(f"  Output shape: {spec.shape}")
            print(f"  Time step: {time_step:.6f} s")

            # Add to results for comparison
            results["multi_resolution"] = {
                "init_time_ms": init_time * 1000,
                "time_ms": comp_time * 1000,
                "shape": spec.shape,
                "spectrogram": spec,
            }

            # Updated summary with multi-resolution
            print(f"\n{'=' * 70}")
            print("Updated Performance Summary (including Multi-Resolution)")
            print(f"{'=' * 70}")
            print(f"{'Method':<30} {'Time (ms)':<15} {'Speedup vs Librosa':<20}")
            print(f"{'-' * 70}")

            librosa_time = results.get("librosa", {}).get("time_ms", None)
            for method, method_data in results.items():
                time_ms = method_data["time_ms"]
                if librosa_time:
                    speedup = librosa_time / time_ms
                    speedup_str = f"{speedup:.2f}x"
                else:
                    speedup_str = "N/A"
                print(f"{method:<30} {time_ms:<15.2f} {speedup_str:<20}")


def test_reverb_augmentation(config: Config, n_samples: int = 1) -> None:
    """Test reverberation augmentation with audio playback and visualization.

    Args:
        config: Configuration object
        n_samples: Number of audio samples to test
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from src.data.augmentation import apply_reverb

    print("\n" + "=" * 70)
    print("Reverberation Augmentation Test")
    print("=" * 70)

    # Create AudioDataset instance
    dataset = AudioDataset(config.data, dataset_type="train_curated")

    print(f"\nDataset size: {len(dataset)} samples")
    print(f"Testing {n_samples} sample(s) with reverb augmentation")

    # Reverb parameters
    rt60_range = (0.5, 2.0)  # Reverberation time: 0.5-2.0 seconds
    dry_wet_mix = 0.5  # 50% wet signal

    print("\nReverb parameters:")
    print(f"  RT60 range: {rt60_range[0]}-{rt60_range[1]} seconds")
    print(f"  Dry/wet mix: {(1 - dry_wet_mix) * 100:.0f}% dry, {dry_wet_mix * 100:.0f}% wet")

    for i in range(min(n_samples, len(dataset))):
        # Find a sample with a recognizable label
        target_labels = ["Acoustic_guitar", "Piano", "Violin_or_fiddle", "Flute", "Trumpet"]
        sample_found = False

        for idx in range(len(dataset)):
            row = dataset.metadata.iloc[idx]
            labels = row["labels"]
            if any(label in labels for label in target_labels):
                sample_found = True
                break

        if not sample_found:
            # Fall back to first sample
            idx = i
            row = dataset.metadata.iloc[idx]

        filename = row["fname"]
        labels = row["labels"]

        print(f"\n{'=' * 70}")
        print(f"Sample {i + 1}: {filename}")
        print(f"Labels: {labels}")
        print(f"{'=' * 70}")

        # Load audio waveform
        waveform, sample_rate = dataset.audio_loader.load_audio(filename)

        # Limit to first 5 seconds for reasonable test
        max_samples = int(5.0 * sample_rate)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
            print(f"  Truncated to {len(waveform) / sample_rate:.1f} seconds")

        print(f"  Original waveform shape: {waveform.shape}")
        print(f"  Sample rate: {sample_rate} Hz")

        # Generate reverb with fixed seed for reproducibility
        rng = np.random.default_rng(42)

        print("\n  Applying reverberation...")
        t0 = time.time()
        reverbed_waveform = apply_reverb(
            waveform=waveform,
            sample_rate=sample_rate,
            rt60_range=rt60_range,
            dry_wet_mix=dry_wet_mix,
            rng=rng,
        )
        print(f"  Reverberation time: {1000 * (time.time() - t0):.2f} ms")
        print(f"  Reverbed waveform shape: {reverbed_waveform.shape}")

        # Generate impulse response for visualization
        from src.data.augmentation import _generate_room_impulse_response

        rng_viz = np.random.default_rng(42)
        impulse_response = _generate_room_impulse_response(
            sample_rate=sample_rate,
            rt60_range=rt60_range,
            rng=rng_viz,
        )

        print(
            f"  Impulse response length: {len(impulse_response)} samples ({len(impulse_response) / sample_rate:.2f}s)"
        )

        # Create visualization
        _fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Plot 1: Impulse response
        time_ir = np.arange(len(impulse_response)) / sample_rate
        axes[0].plot(time_ir, impulse_response, linewidth=0.5, color="darkblue")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title(
            f"Room Impulse Response (RT60 ≈ {len(impulse_response) / sample_rate:.2f}s)"
        )
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Original waveform
        time_orig = np.arange(len(waveform)) / sample_rate
        axes[1].plot(time_orig, waveform, linewidth=0.5, color="green")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Amplitude")
        axes[1].set_title(f"Original Audio: {labels}")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([-1.1, 1.1])

        # Plot 3: Reverbed waveform
        time_rev = np.arange(len(reverbed_waveform)) / sample_rate
        axes[2].plot(time_rev, reverbed_waveform, linewidth=0.5, color="orangered")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Amplitude")
        axes[2].set_title(
            f"Reverbed Audio (dry/wet: {(1 - dry_wet_mix) * 100:.0f}/{dry_wet_mix * 100:.0f}%)"
        )
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([-1.1, 1.1])

        plt.tight_layout()
        plt.show()

        # Save temporary files for playback
        import tempfile

        import soundfile as sf

        with (
            tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_orig,
            tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_rev,
        ):
            # Write temporary audio files
            sf.write(tmp_orig.name, waveform, sample_rate)
            sf.write(tmp_rev.name, reverbed_waveform, sample_rate)

            # Play original
            print("\n  Playing ORIGINAL audio...")
            result = play_audio_file(Path(tmp_orig.name))
            if result == "quit":
                print("\nPlayback stopped by user")
                return

            # Play reverbed
            print("\n  Playing REVERBED audio...")
            result = play_audio_file(Path(tmp_rev.name))
            if result == "quit":
                print("\nPlayback stopped by user")
                return

        print(f"\n{'=' * 70}")
        print("Reverb test completed successfully!")
        print(f"{'=' * 70}")


def main() -> None:
    """Main development entry point."""
    parser = argparse.ArgumentParser(description="Development and testing tools")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--n-samples", type=int, default=3, help="Number of audio samples to test")
    parser.add_argument(
        "--test",
        type=str,
        choices=["audio", "signal", "spectrogram", "benchmark", "reverb"],
        default="signal",
        help="Which test to run: audio (dataset loading), signal (envelope patterns), spectrogram (comparison), benchmark (optimized methods), reverb (reverberation augmentation)",
    )
    parser.add_argument(
        "--test-gpu",
        action="store_true",
        help="Test GPU acceleration in benchmark mode (requires PyTorch with MPS)",
    )
    args = parser.parse_args()

    # Load configuration using new Config.from_yaml method
    config = Config.from_yaml(args.config)

    print("Configuration loaded successfully!")
    print(f"Project: {config.project.name}")
    print(f"Data directory: {config.data.base_dir}")
    print(f"Sample rate: {config.data.sample_rate} Hz")
    print(f"Spectrogram bands: {config.spectrogram.n_bands}")

    # Run selected test
    if args.test == "audio":
        test_audio_loading(config, n_samples=args.n_samples)
    elif args.test == "signal":
        signal_tools.dev_envelope_pattern()
    elif args.test == "spectrogram":
        test_spectrogram_comparison(config, n_samples=args.n_samples)
    elif args.test == "benchmark":
        test_spectrogram_benchmark(config, n_samples=args.n_samples, test_gpu=args.test_gpu)
    elif args.test == "reverb":
        test_reverb_augmentation(config, n_samples=args.n_samples)


if __name__ == "__main__":
    main()
