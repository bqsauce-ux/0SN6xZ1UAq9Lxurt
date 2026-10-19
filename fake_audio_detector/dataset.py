
from pathlib import Path

import torch
from torch.utils.data import Dataset
import librosa

from features import extract_features, TARGET_SR, MIN_DURATION


class FakeAudioDataset(Dataset):

    def __init__(self, real_dir, fake_dir):

        self.samples = []

        self.real_dir = Path(real_dir)
        self.fake_dir = Path(fake_dir)

        extensions = {".wav", ".mp3", ".flac", ".ogg"}

        # Find audio files
        real_files = [
            f for f in self.real_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in extensions
        ]

        fake_files = [
            f for f in self.fake_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in extensions
        ]

        print("\nReal audio directory:")
        print(self.real_dir.resolve())

        print("\nFake audio directory:")
        print(self.fake_dir.resolve())

        print(f"\nOriginal real files: {len(real_files)}")
        print(f"Original fake files: {len(fake_files)}")

        # Validate audio duration after silence trimming
        self._add_valid_samples(real_files, label=0)
        self._add_valid_samples(fake_files, label=1)

        # Report filtering results
        valid_real = sum(label == 0 for _, label in self.samples)
        valid_fake = sum(label == 1 for _, label in self.samples)

        print("\nDataset filtering results:")
        print(f"Valid real samples: {valid_real}")
        print(f"Valid fake samples: {valid_fake}")
        print(f"Skipped short/unreadable files: {self.skipped}")
        print(f"Total valid samples: {len(self.samples)}")

    def _add_valid_samples(self, files, label):

        if not hasattr(self, "skipped"):
            self.skipped = 0

        for audio_file in files:
            try:
                # Load audio using the same sample rate as feature extraction
                audio, _ = librosa.load(
                    audio_file,
                    sr=TARGET_SR,
                    mono=True
                )

                # Match the silence-trimming step in features.py
                audio, _ = librosa.effects.trim(
                    audio,
                    top_db=30
                )

                duration = len(audio) / TARGET_SR

                if duration < MIN_DURATION:
                    self.skipped += 1
                    print(f"Skipping short audio ({duration:.3f}s): {audio_file}")
                    continue

                self.samples.append((audio_file, label))

            except Exception as e:
                self.skipped += 1
                print(f"Skipping unreadable audio: {audio_file} ({e})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        audio_file, label = self.samples[index]

        features = extract_features(audio_file)

        features = torch.tensor(
            features,
            dtype=torch.float32
        )

        # [128, 400] -> [1, 128, 400]
        features = features.unsqueeze(0)

        return features, label
