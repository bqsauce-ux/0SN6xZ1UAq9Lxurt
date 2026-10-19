import torch
import torchaudio

from transformers import WavLMModel, Wav2Vec2FeatureExtractor


class SpeakerEncoder:

    def __init__(self):

        print("Loading pretrained speaker encoder...")

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus"
        )

        self.model = WavLMModel.from_pretrained(
            "microsoft/wavlm-base-plus"
        )

        self.model.eval()

        print("Speaker encoder loaded successfully.")


    def encode(self, wav_path):

        waveform, sample_rate = torchaudio.load(wav_path)

        # Convert stereo → mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # WavLM expects 16 kHz
        if sample_rate != 16000:

            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            )

            waveform = resampler(waveform)

        waveform = waveform.squeeze(0)

        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = self.model(
                inputs.input_values
            )

        # Mean-pool across time
        embedding = outputs.last_hidden_state.mean(
            dim=1
        )

        # Normalize
        embedding = torch.nn.functional.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding.squeeze(0)


if __name__ == "__main__":

    encoder = SpeakerEncoder()

    print(
        "Speaker encoder loaded successfully."
    )