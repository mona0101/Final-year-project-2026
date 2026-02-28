import torch
from PIL import Image, ImageFilter
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms as T
import augly.audio as audaugs
import random
import librosa

# --- Constants ---
IMAGENET_PIXEL_MEAN = [123.675, 116.280, 103.530]
IMAGENET_PIXEL_STD = [58.395, 57.12, 57.375]
SR = 16000              
SEGMENT_LENGTH = 0.25

# --- Image Helper Functions ---
def salt_and_pepper_noise(image, salt_vs_pepper=0.5, amount=0.01):
    img_array = image.numpy().copy()
    num_salt = np.ceil(amount * img_array.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * img_array.size * (1. - salt_vs_pepper))

    coords = [np.random.randint(0, i, int(num_salt)) for i in img_array.shape]
    img_array[tuple(coords)] = 1

    coords = [np.random.randint(0, i, int(num_pepper)) for i in img_array.shape]
    img_array[tuple(coords)] = 0

    return torch.from_numpy(img_array)

# --- Audio Log-Mel Helper Functions ---
def add_white_noise(y, noise_factor=0.02):
    noise = np.random.randn(len(y)).astype(np.float32)
    return (y + noise_factor * noise).astype(np.float32)

def random_gain(y, min_gain=0.8, max_gain=1.2):
    gain = np.random.uniform(min_gain, max_gain)
    return (y * gain).astype(np.float32)

def time_shift(y, shift_max=0.1, sr=SR):
    shift = int(np.random.uniform(-shift_max, shift_max) * sr)
    return np.roll(y, shift).astype(np.float32)

def time_stretch_fixlen(y, rate_min=0.9, rate_max=1.1, sr=SR, segment_length=SEGMENT_LENGTH):
    rate = np.random.uniform(rate_min, rate_max)
    y_stretch = librosa.effects.time_stretch(y, rate=rate)
    target_len = int(segment_length * sr)
    
    if len(y_stretch) < target_len:
        y_stretch = np.pad(y_stretch, (0, target_len - len(y_stretch)))
    else:
        y_stretch = y_stretch[:target_len]
    
    max_val = np.max(np.abs(y_stretch))
    if max_val > 0:
        y_stretch = y_stretch / max_val
    return y_stretch.astype(np.float32)

# --- Audio Augmentation Function ---
def get_audio_transform(args, is_training, feature_type):
    if not is_training:
        return lambda waveform, sr: (waveform, sr)

    if feature_type == 'mfcc':
        return audaugs.Compose([
            audaugs.AddBackgroundNoise(),
            audaugs.Harmonic(),
            audaugs.OneOf([
                audaugs.PitchShift(),
                audaugs.Clicks(),
                audaugs.ToMono(),
                audaugs.ChangeVolume(volume_db=10.0, p=0.5)
            ]),
        ])

    elif feature_type == 'logmel':
        def logmel_augment(waveform, sr):
            # STOCHASTIC APPROACH: Randomly choose ONE augmentation or original
            choice = random.choice(["original", "noise", "gain", "shift", "stretch"])
            
            if choice == "noise":
                waveform = add_white_noise(waveform, noise_factor=0.02)
            elif choice == "gain":
                waveform = random_gain(waveform, min_gain=0.8, max_gain=1.2)
            elif choice == "shift":
                waveform = time_shift(waveform, shift_max=0.1, sr=sr)
            elif choice == "stretch":
                waveform = time_stretch_fixlen(waveform, sr=sr, segment_length=SEGMENT_LENGTH)
            
            return waveform.astype(np.float32), sr
        
        return logmel_augment

    return lambda waveform, sr: (waveform, sr)

# --- RF Helper Functions ---
def rf_time_shift(tensor, max_shift_ratio=0.15):
    x = tensor.clone()
    _, H, W = x.shape
    max_shift = int(W * max_shift_ratio)
    if max_shift < 1: return x
    shift = random.randint(-max_shift, max_shift)
    return torch.roll(x, shifts=shift, dims=2)

def rf_time_mask(tensor, max_mask_ratio=0.25):
    x = tensor.clone()
    _, H, W = x.shape
    mask_w = int(W * random.uniform(0.05, max_mask_ratio))
    if mask_w < 1: return x
    start_x = random.randint(0, max(0, W - mask_w))
    x[:, :, start_x:start_x+mask_w] = 0.0
    return x

def rf_freq_mask(tensor, max_mask_ratio=0.25):
    x = tensor.clone()
    _, H, W = x.shape
    mask_h = int(H * random.uniform(0.05, max_mask_ratio))
    if mask_h < 1: return x
    start_y = random.randint(0, max(0, H - mask_h))
    x[:, start_y:start_y+mask_h, :] = 0.0
    return x

def rf_add_gaussian_noise(tensor, std=0.05):
    return tensor + torch.randn_like(tensor) * std

def rf_gaussian_blur(tensor, radius=1.0):
    pil_img = T.ToPILImage()(tensor)
    pil_blur = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    t = T.ToTensor()(pil_blur)
    return (t - 0.5) / 0.5

def separate_rf_augmentation(x):
    choice = random.choice(["original", "time_shift", "time_mask", "freq_mask", "noise", "blur"])
    separate_rf_augmentation.last_choice = choice   

    if choice == "time_shift": x = rf_time_shift(x)
    elif choice == "time_mask": x = rf_time_mask(x)
    elif choice == "freq_mask": x = rf_freq_mask(x)
    elif choice == "noise": x = rf_add_gaussian_noise(x)
    elif choice == "blur": x = rf_gaussian_blur(x)
    return x

# --- Master Transform Functions ---
def get_normalize():
    return transforms.Normalize(
        mean=torch.Tensor(IMAGENET_PIXEL_MEAN) / 255.0,
        std=torch.Tensor(IMAGENET_PIXEL_STD) / 255.0,
    )

def get_img_transform(args, is_training, augment=True):
    train_crop_size = getattr(args, 'train_crop_size', args.crop_size)
    test_crop_size = getattr(args, 'test_crop_size', args.crop_size)
    interpolation = Image.BILINEAR if getattr(args, 'interpolation', None) == 'bilinear' else Image.BICUBIC
    normalize = get_normalize()

    if is_training and augment:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(train_crop_size, interpolation=interpolation),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            transforms.GaussianBlur(3),
            lambda x: salt_and_pepper_noise(x, salt_vs_pepper=0.45, amount=0.02),
            normalize,
        ])
    
    target_size = train_crop_size if is_training else test_crop_size
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(target_size, interpolation=interpolation),
        normalize,
    ])

def get_rf_transform(args, is_training, augment=False):
    train_crop_size = getattr(args, 'train_crop_size', args.crop_size)
    test_crop_size = getattr(args, 'test_crop_size', args.crop_size)
    interpolation = Image.BILINEAR if getattr(args, 'interpolation', None) == 'bilinear' else Image.BICUBIC
    normalize = get_normalize()
    target_size = train_crop_size if is_training else test_crop_size

    if is_training and augment:
       return transforms.Compose([
         transforms.Resize(target_size, interpolation=interpolation),
         transforms.ToTensor(),
         lambda x: separate_rf_augmentation(x),
         normalize,
        ])

    return transforms.Compose([
        transforms.Resize(target_size, interpolation=interpolation),
        transforms.ToTensor(),
        normalize,
    ])
