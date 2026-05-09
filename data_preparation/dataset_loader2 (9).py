import os
import re
import librosa
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

def numerical_sort(value):
    parts = re.split(r'(\d+)', value)
    parts[1::2] = map(int, parts[1::2])
    return parts

class DroneFusionDataset(Dataset):
    def __init__(
        self,
        audio_root,
        video_root,
        rf_root,
        dataset_type='Train',
        transform=None,
        n_mfcc=40,
        n_fft=2048,
        hop_length=512,
        duration=10,
        segment_length=0.25,
        audio_feature_type='mfcc',
        audio_sr=None,
        n_mels=64,  # Added: allows easy change of Mel frequency bins
        modalities=['audio', 'video', 'rf']
    ):
        self.audio_root = audio_root
        self.video_root = video_root
        self.rf_root = rf_root
        self.dataset_type = dataset_type.lower()
        self.transform = transform or {}
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.segment_length = segment_length
        self.audio_feature_type = audio_feature_type.lower()
        self.audio_sr = audio_sr
        self.n_mels = n_mels  # Added: number of Mel bins for log-Mel
        self.modalities = [m.lower() for m in modalities]
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        categories = ['Drone', 'Background']
        video_folder_name = 'Images_Extracted' if self.dataset_type == 'train' else 'Images'
        rf_folder_name = 'Images_Spectrograms' if self.dataset_type == 'train' else 'Images'

        for category in categories:
            audio_category_path = os.path.join(self.audio_root, category)
            video_category_path = os.path.join(self.video_root, category)
            rf_category_path = os.path.join(self.rf_root, category)

            if not os.path.exists(audio_category_path):
                continue

            scenarios = sorted(os.listdir(audio_category_path), key=numerical_sort)
            for scenario in scenarios:
                audio_scenario_path = os.path.join(audio_category_path, scenario)
                video_scenario_path = os.path.join(video_category_path, scenario, video_folder_name)
                rf_scenario_path = os.path.join(rf_category_path, scenario, rf_folder_name)

                audio_files = sorted(
                    [f for f in os.listdir(audio_scenario_path) if f.endswith('.wav')],
                    key=numerical_sort
                )

                for audio_file in audio_files:
                    audio_file_number = re.findall(r'\d+', audio_file)[0]
                    audio_path = os.path.join(audio_scenario_path, audio_file)
                    label = 1 if category == "Drone" else 0

                    num_segments = int(self.duration / self.segment_length)

                    for segment in range(num_segments):
                        start_time = segment * self.segment_length
                        end_time = (segment + 1) * self.segment_length

                        video_frame_start = int(start_time * 28) + 1
                        video_frame_end = int(end_time * 28)
                        rf_frame_start = int(start_time * 4) + 1
                        rf_frame_end = int(end_time * 4)

                        video_frames = [
                            os.path.join(video_scenario_path, f"{audio_file_number}_frame_{i}.jpg")
                            for i in range(max(1, video_frame_start), min(video_frame_end, 300) + 1)
                        ]
                        rf_frames = [
                            os.path.join(rf_scenario_path, f"{audio_file_number}_frame_{i}.jpg")
                            for i in range(max(1, rf_frame_start), min(rf_frame_end, 40) + 1)
                        ]

                        samples.append({
                            'audio_path': audio_path,
                            'video_frame_paths': video_frames,
                            'rf_frame_paths': rf_frames,
                            'label': label,
                            'start': start_time,
                            'end': end_time
                        })
        return samples

    def _extract_logmel(self, y, sr):
        """
        Extract log-Mel spectrogram with z-score normalization.
        Automatically calculates target_frames based on segment_length and hop_length.
        """
        # Added: automatic target_frames calculation
        target_frames = int(np.ceil(self.segment_length / (self.hop_length / sr)))

        # Compute Mel spectrogram using user-specified parameters
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=1024,          # Using your preferred n_fft
            hop_length=138,      # Using your preferred hop_length
            n_mels=self.n_mels,  # Using user-defined n_mels
            power=2.0
        )

        # Convert to log scale
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Z-score normalization
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

        # Pad or truncate to match calculated target_frames
        if log_mel.shape[1] < target_frames:
            pad_width = target_frames - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0,0),(0,pad_width)), mode='constant')
        else:
            log_mel = log_mel[:, :target_frames]

        return torch.from_numpy(log_mel).unsqueeze(0).float()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # --- Audio ---
        audio_tensor = torch.empty(0)
        if 'audio' in self.modalities:
            sr = self.audio_sr if self.audio_sr is not None else (22050 if self.audio_feature_type == 'logmel' else 44100)
            waveform, _ = librosa.load(sample['audio_path'], sr=sr, offset=sample['start'], duration=self.segment_length)

            if 'audio' in self.transform:
                waveform, sr = self.transform['audio'](waveform, sr)

            if self.audio_feature_type == 'logmel':
                max_val = np.max(np.abs(waveform))
                if max_val > 0: waveform = waveform / max_val
                audio_tensor = self._extract_logmel(waveform, sr)
            else:  # MFCC
                waveform = np.asfortranarray(waveform[::3])
                mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
                pad_width = max(0, self.n_mfcc - mfcc.shape[1])
                mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
                audio_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

        # --- Video ---
        video_tensor = torch.empty(0)
        if 'video' in self.modalities:
            video_frames = [Image.open(p).convert("RGB") for p in sample['video_frame_paths'] if os.path.exists(p)]
            if 'video' in self.transform and video_frames:
                video_tensor = torch.stack([self.transform['video'](f) for f in video_frames])

        # --- RF ---
        rf_tensor = torch.empty(0)
        if 'rf' in self.modalities:
            rf_frames = [Image.open(p).convert("RGB") for p in sample['rf_frame_paths'] if os.path.exists(p)]
            if 'rf' in self.transform and rf_frames:
                rf_tensor = torch.stack([self.transform['rf'](f) for f in rf_frames])

        return audio_tensor, video_tensor, rf_tensor, sample['label']


def get_loader(dataset, batch_size=4, is_train=True, num_workers=0, pin_memory=False):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers, pin_memory=pin_memory)
