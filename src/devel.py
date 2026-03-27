"""Development and testing module for audio dataset functionality."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.utils import load_config

from . import data, signal_tools


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


def test_audio_loading(config: dict, n_samples: int = 3) -> None:
    """Test audio dataset loading and display info for n samples.

    Args:
        config: Configuration dictionary
        n_samples: Number of samples to test
    """
    print("\n" + "=" * 60)
    print("Testing AudioDataset")
    print("=" * 60)

    # Create dataset config and AudioDataset instance
    dataset_config = data.AudioDatasetConfig.from_dict(config["data"])
    dataset = data.AudioDataset(dataset_config, dataset_type="train_curated")

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
        waveform, sr = dataset.audio_loader.load_audio(filename)
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


def main() -> None:
    """Main development entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--n-samples", type=int, default=3, help="Number of audio samples to test"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print("Loaded config:\n", json.dumps(config, indent=4))

    # Test audio dataset loading
    test_audio_loading(config, n_samples=args.n_samples)

    # Uncomment to test signal processing tools
    # signal_tools.dev_gaussian()
    # signal_tools.dev_super_gaussian()


if __name__ == "__main__":
    main()
