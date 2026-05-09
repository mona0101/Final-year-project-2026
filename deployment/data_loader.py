import os
import re
import zipfile
import tempfile
import shutil
import torch
import numpy as np
import librosa
from PIL import Image
import torchvision.transforms as transforms

# ============================================================
# Constants
# ============================================================
SAMPLE_RATE = 22050
SEGMENT_LENGTH = 0.25
HOP_LENGTH = 138
N_MELS = 64
CROP_SIZE = 112

# Image normalization (ImageNet standards)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Image transform: resize -> to tensor -> normalize
image_transform = transforms.Compose([
    transforms.Resize((CROP_SIZE, CROP_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# ============================================================
# Helper Functions
# ============================================================
def numerical_sort(value):
    """Sort filenames with natural number order"""
    parts = re.split(r'(\d+)', value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def extract_zip(zip_file, extract_to):
    """Extract zip file and return the actual folder path"""
    with zipfile.ZipFile(zip_file, 'r') as zf:
        zf.extractall(extract_to)
    items = os.listdir(extract_to)
    if len(items) == 1 and os.path.isdir(os.path.join(extract_to, items[0])):
        return os.path.join(extract_to, items[0])
    return extract_to

def find_audio_files(folder):
    """Find all .wav files"""
    files = []
    for root, dirs, filenames in os.walk(folder):
        for f in filenames:
            if f.endswith('.wav'):
                files.append(os.path.join(root, f))
    return sorted(files, key=numerical_sort)

def find_image_files(folder):
    """Find all .jpg / .png files"""
    files = []
    for root, dirs, filenames in os.walk(folder):
        for f in filenames:
            if f.endswith('.jpg') or f.endswith('.png'):
                files.append(os.path.join(root, f))
    return sorted(files, key=numerical_sort)

def get_audio_id(filepath):
    """Extract audio ID from filename (e.g., '15.wav' -> '15')"""
    return os.path.splitext(os.path.basename(filepath))[0]

def cleanup_temp_dir(temp_dir):
    """Remove temporary directory"""
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

# ============================================================
# Audio Preprocessing
# ============================================================
def extract_logmel(waveform, sr):
    """Extract log-Mel spectrogram from waveform -> [1, 64, 40]"""
    target_frames = int(np.ceil(SEGMENT_LENGTH / (HOP_LENGTH / sr)))

    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_fft=1024, hop_length=HOP_LENGTH,
        n_mels=N_MELS, power=2.0
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

    if log_mel.shape[1] < target_frames:
        pad_width = target_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
    else:
        log_mel = log_mel[:, :target_frames]

    return torch.from_numpy(log_mel).unsqueeze(0).float()


def preprocess_audio(audio_path, segment_index=0):
    """
    Load and preprocess a single 0.25-second segment -> tensor [1, 64, 40].

    FIXED: uses segment_index to compute the correct time offset so that
    each of the 40 segments gets its own audio slice, not always offset=0.
    """
    offset = segment_index * SEGMENT_LENGTH
    waveform, _ = librosa.load(
        audio_path, sr=SAMPLE_RATE, offset=offset, duration=SEGMENT_LENGTH
    )

    # Normalize
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val

    return extract_logmel(waveform, SAMPLE_RATE)


# ============================================================
# Video Preprocessing
# ============================================================
def preprocess_video_frames(frame_paths):
    """Process 7 video frames -> tensor [7, 3, 112, 112]"""
    frames = []
    for path in frame_paths[:7]:
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
            frames.append(image_transform(img))
    return torch.stack(frames) if frames else None


# ============================================================
# RF Preprocessing
# ============================================================
def preprocess_rf_image(image_path):
    """
    Process RF grayscale spectrogram -> tensor [3, 112, 112].

    FIXED:
    - Removed the extra .unsqueeze(0) that was creating shape [1,3,112,112].
      app.py adds the batch dim with its own .unsqueeze(0), so the output
      going into the model is [B, 3, 112, 112] as ResNet18 expects.
    - RF images are grayscale spectrograms but ResNet18 (pretrained on
      ImageNet) requires 3-channel input, so we replicate channels via
      .convert("RGB").
    """
    img = Image.open(image_path).convert("RGB")
    return image_transform(img)   # shape: [3, 112, 112]


# ============================================================
# Main Data Loading Function
# ============================================================
def load_uploaded_data(audio_zip, video_zip, rf_zip):
    """
    Extract uploaded ZIP files and return:
      - audio_files : list of audio file paths
      - video_files : list of video frame paths
      - rf_files    : list of RF frame paths
      - temp_dir    : temporary directory path (for cleanup)
    """
    temp_dir = tempfile.mkdtemp()

    if audio_zip:
        audio_folder = extract_zip(audio_zip, os.path.join(temp_dir, "audio"))
        audio_files  = find_audio_files(audio_folder)
    else:
        audio_files = []

    if video_zip:
        video_folder = extract_zip(video_zip, os.path.join(temp_dir, "video"))
        video_files  = find_image_files(video_folder)
    else:
        video_files = []

    if rf_zip:
        rf_folder = extract_zip(rf_zip, os.path.join(temp_dir, "rf"))
        rf_files  = find_image_files(rf_folder)
    else:
        rf_files = []

    return audio_files, video_files, rf_files, temp_dir


def get_audio_segments(audio_id, video_files, rf_files):
    """
    Get matching video and RF frames for a given audio ID.
    Returns (matching_video, matching_rf, num_segments).

    Segment count = min(video_frames // 7, rf_frames, 40).
    With 300 video frames and 40 RF frames:
      min(300 // 7, 40, 40) = min(42, 40, 40) = 40  ✓
    """
    matching_video = [f for f in video_files if f"{audio_id}_frame_" in os.path.basename(f)]
    matching_video = sorted(matching_video, key=numerical_sort)

    matching_rf = [f for f in rf_files if f"{audio_id}_frame_" in os.path.basename(f)]
    matching_rf  = sorted(matching_rf, key=numerical_sort)

    num_segments = min(len(matching_video) // 7, len(matching_rf), 40)

    return matching_video, matching_rf, num_segments


def get_segment_data(video_frames, rf_frames, segment_index):
    """
    Get the 7 video frames and 1 RF frame for a specific segment index.
    """
    video_segment = video_frames[segment_index * 7:(segment_index + 1) * 7]
    rf_frame = rf_frames[segment_index] if segment_index < len(rf_frames) else None
    return video_segment, rf_frame


# ============================================================
# Batch Preprocessing — all 40 segments at once
# ============================================================
def preprocess_all_segments(audio_path, video_frames, rf_frames, num_segments):
    """
    Preprocess every segment for a single audio file.

    Returns a list of dicts (one per segment), each containing:
      'audio'       : tensor [1, 64, 40]
      'video'       : tensor [7, 3, 112, 112]  or None
      'rf'          : tensor [3, 112, 112]       or None
      'video_paths' : list of 7 frame file paths
      'rf_path'     : single RF frame file path or None
    """
    segments = []
    for i in range(num_segments):
        audio_tensor = preprocess_audio(audio_path, segment_index=i)

        video_segment_paths = video_frames[i * 7:(i + 1) * 7]
        video_tensor = preprocess_video_frames(video_segment_paths)

        rf_path   = rf_frames[i] if i < len(rf_frames) else None
        rf_tensor = preprocess_rf_image(rf_path) if rf_path else None

        segments.append({
            'audio':       audio_tensor,
            'video':       video_tensor,
            'rf':          rf_tensor,
            'video_paths': video_segment_paths,
            'rf_path':     rf_path,
        })
    return segments