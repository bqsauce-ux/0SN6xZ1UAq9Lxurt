import librosa
import numpy as np
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

TARGET_SR = 16000

MAX_DURATION = 10

MIN_DURATION = 0.5

N_MELS = 128

N_FFT = 1024

HOP_LENGTH = 256

MAX_FRAMES = 400


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(audio_file):

    audio_file = Path(audio_file)

    # -----------------------------------------------------
    # Load audio
    # -----------------------------------------------------

    audio, sr = librosa.load(
        audio_file,
        sr=TARGET_SR,
        mono=True
    )

    # -----------------------------------------------------
    # Trim silence
    # -----------------------------------------------------

    audio, _ = librosa.effects.trim(
        audio,
        top_db=30
    )

    # -----------------------------------------------------
    # Check minimum duration
    # -----------------------------------------------------

    duration = len(audio) / TARGET_SR

    if duration < MIN_DURATION:

        raise ValueError(
            f"Audio too short: {audio_file}"
        )

    # -----------------------------------------------------
    # Limit audio to MAX_DURATION
    # -----------------------------------------------------

    max_samples = int(
        MAX_DURATION * TARGET_SR
    )

    if len(audio) > max_samples:

        audio = audio[:max_samples]

    # -----------------------------------------------------
    # Normalize audio
    # -----------------------------------------------------

    max_amplitude = np.max(
        np.abs(audio)
    )

    if max_amplitude > 0:

        audio = (
            audio / max_amplitude
        )

    # -----------------------------------------------------
    # Mel spectrogram
    # -----------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    # -----------------------------------------------------
    # Convert to dB
    # -----------------------------------------------------

    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # -----------------------------------------------------
    # Standardize
    # -----------------------------------------------------

    mean = np.mean(mel)

    std = np.std(mel)

    if std > 0:

        mel = (
            mel - mean
        ) / std

    # =====================================================
    # FORCE EXACTLY 400 TIME FRAMES
    # =====================================================

    if mel.shape[1] < MAX_FRAMES:

        padding_width = (
            MAX_FRAMES - mel.shape[1]
        )

        mel = np.pad(
            mel,
            (
                (0, 0),
                (0, padding_width)
            ),
            mode="constant",
            constant_values=0
        )

    elif mel.shape[1] > MAX_FRAMES:

        mel = mel[
            :,
            :MAX_FRAMES
        ]

    # -----------------------------------------------------
    # Verify shape
    # -----------------------------------------------------

    if mel.shape != (
        N_MELS,
        MAX_FRAMES
    ):

        raise RuntimeError(
            f"Unexpected feature shape: {mel.shape}"
        )

    return mel.astype(
        np.float32
    )