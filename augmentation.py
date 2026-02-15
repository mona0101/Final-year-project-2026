import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import augly.audio as audaugs
import random
from PIL import Image, ImageFilter
import torchvision.transforms as T
import librosa

# standard normalization values used when training models (most common for PyTorch models like ResNet, VGG, EfficientNet)
IMAGENET_PIXEL_MEAN = [123.675, 116.280, 103.530]
IMAGENET_PIXEL_STD = [58.395, 57.12, 57.375]
SR = 16000              # Sampling rate
SEGMENT_LENGTH = 0.25

def salt_and_pepper_noise(image, salt_vs_pepper=0.5, amount=0.01):
    # image is a tensor (C, H, W)
    img_array = image.numpy().copy()

    # Calculate number of pixels to alter
    num_salt = np.ceil(amount * img_array.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * img_array.size * (1. - salt_vs_pepper))

    # Add Salt (White dots)
    # We generate coordinates for all 3 dimensions: Channel, Height, Width
    coords = [np.random.randint(0, i, int(num_salt)) for i in img_array.shape]
    img_array[tuple(coords)] = 1

    # Add Pepper (Black dots)
    coords = [np.random.randint(0, i, int(num_pepper)) for i in img_array.shape]
    img_array[tuple(coords)] = 0

    return torch.from_numpy(img_array)

# audio log mel augmentation helper functions

def add_white_noise(y, noise_factor=0.02):
    noise = np.random.randn(len(y)).astype(np.float32)
    return (y + noise_factor * noise).astype(np.float32)

def random_gain(y, min_gain=0.8, max_gain=1.2):
    gain = np.random.uniform(min_gain, max_gain)
    return (y * gain).astype(np.float32)

def time_shift(y, shift_max=0.1, sr=SR):
    """
    shift_max = 0.1 sec
    """
    shift = int(np.random.uniform(-shift_max, shift_max) * sr)
    return np.roll(y, shift).astype(np.float32)

def time_stretch_fixlen(y, rate_min=0.9, rate_max=1.1,
                        sr=SR, segment_length=SEGMENT_LENGTH):
    rate = np.random.uniform(rate_min, rate_max)
    y_stretch = librosa.effects.time_stretch(y, rate=rate)

    target_len = int(segment_length * sr)
    if len(y_stretch) < target_len:
        pad_width = target_len - len(y_stretch)
        y_stretch = np.pad(y_stretch, (0, pad_width))
    else:
        y_stretch = y_stretch[:target_len]

    # normalization
    max_val = np.max(np.abs(y_stretch))
    if max_val > 0:
        y_stretch = y_stretch / max_val

    return y_stretch.astype(np.float32)


# audio augmentation function

def get_audio_transform(args, is_training, feature_type):

    if not is_training:
        return lambda waveform, sr: (waveform, sr) # No augmentations

    # MFCC augmentation
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


    # LOG-MEL augmentation
    elif feature_type == 'logmel':

      def logmel_augment(waveform, sr):

        # Apply ALL 4 augmentations
        waveform = add_white_noise(waveform, noise_factor=0.02)
        waveform = random_gain(waveform, min_gain=0.8, max_gain=1.2)
        waveform = time_shift(waveform, shift_max=0.1, sr=sr)
        waveform = time_stretch_fixlen(
            waveform,
            rate_min=0.9,
            rate_max=1.1,
            sr=sr,
            segment_length=SEGMENT_LENGTH
        )

        return waveform.astype(np.float32), sr

    return logmel_augment


# image augmentation function
def get_img_transform(args, is_training):

    train_crop_size = getattr(args, 'train_crop_size', args.crop_size)
    test_scale = getattr(args, 'test_scale', args.scale_size)
    test_crop_size = getattr(args, 'test_crop_size', args.crop_size)

    interpolation = Image.BICUBIC
    if getattr(args, 'interpolation', None) and  args.interpolation == 'bilinear':
        interpolation = Image.BILINEAR

    normalize = get_normalize()

    if not is_training: # No augmentation
        ret = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(train_crop_size, interpolation=interpolation),
            ]
        )
    else:
        ret = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(test_crop_size, interpolation=interpolation),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.4, contrast=0.4),
                transforms.GaussianBlur(3),
                lambda x: salt_and_pepper_noise(x, salt_vs_pepper=0.45, amount=0.02),
            ]
        )
    return ret


# rf augmentation helper functions
def rf_time_shift(tensor, max_shift_ratio=0.15):
    x = tensor.clone()
    _, H, W = x.shape
    max_shift = int(W * max_shift_ratio)
    if max_shift < 1:
        return x
    shift = random.randint(-max_shift, max_shift)
    x = torch.roll(x, shifts=shift, dims=2)
    return x

def rf_time_mask(tensor, max_mask_ratio=0.25):
    x = tensor.clone()
    _, H, W = x.shape
    mask_w = int(W * random.uniform(0.05, max_mask_ratio))
    if mask_w < 1:
        return x
    start_x = random.randint(0, max(0, W - mask_w))
    x[:, :, start_x:start_x+mask_w] = 0.0
    return x

def rf_freq_mask(tensor, max_mask_ratio=0.25):
    x = tensor.clone()
    _, H, W = x.shape
    mask_h = int(H * random.uniform(0.05, max_mask_ratio))
    if mask_h < 1:
        return x
    start_y = random.randint(0, max(0, H - mask_h))
    x[:, start_y:start_y+mask_h, :] = 0.0
    return x

def rf_add_gaussian_noise(tensor, std=0.05):
    x = tensor.clone()
    noise = torch.randn_like(x) * std
    return x + noise

def rf_gaussian_blur(tensor, radius=1.0):
    pil_img = T.ToPILImage()(tensor)
    pil_blur = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    t = T.ToTensor()(pil_blur)
    t = (t - 0.5) / 0.5
    return t


# RF augmentation function
def get_rf_transform(args, is_training, augment=False):

    train_crop_size = getattr(args, 'train_crop_size', args.crop_size)
    test_scale = getattr(args, 'test_scale', args.scale_size)
    test_crop_size = getattr(args, 'test_crop_size', args.crop_size)
    interpolation = Image.BICUBIC

    if getattr(args, 'interpolation', None) and args.interpolation == 'bilinear':
        interpolation = Image.BILINEAR

      # Base transform: resize + to tensor
    if is_training:
        base_transform = transforms.Compose([
            transforms.Resize(train_crop_size, interpolation=interpolation),
            transforms.ToTensor(),
        ])
    else:
        base_transform = transforms.Compose([
            transforms.Resize(test_crop_size, interpolation=interpolation),
            transforms.ToTensor(),
        ])

      # Wrapper function
    def transform_fn(img):
        # Convert PIL image to tensor
        base_tensor = base_transform(img)

        if is_training and augment:
            # Apply all RF augmentations
            aug_list = [
                ("Time Shift",       rf_time_shift(base_tensor)),
                ("Time Mask",        rf_time_mask(base_tensor)),
                ("Freq Mask",        rf_freq_mask(base_tensor)),
                ("Gaussian Noise",   rf_add_gaussian_noise(base_tensor)),
                ("Gaussian Blur",    rf_gaussian_blur(base_tensor)),
            ]
            return base_tensor, aug_list
        else:
            return base_tensor
    return transform_fn



def get_normalize():
    normalize = transforms.Normalize(
        mean=torch.Tensor(IMAGENET_PIXEL_MEAN) / 255.0,
        std=torch.Tensor(IMAGENET_PIXEL_STD) / 255.0,
    )
    return normalize