import cv2
import numpy as np
import random
from skimage.transform import rotate


class ChannelShift:
    def __init__(self, intensity, seed=2022):
        self.name = "ChannelShift"
        assert 1 < intensity < 255, "Set the pixel values to be shifted (1, 255)"
        self.intensity = intensity
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def apply(self, img):
        _, _, ch = img.shape
        img = img.astype(np.float32)
        for i in range(ch):
            img[:, :, i] += self.rng.integers(self.intensity) * self.rng.choice([1, -1])
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)


class Stripes:
    def __init__(self, horizontal, vertical, space, width, intensity):
        self.name = "Stripes"
        self.horizontal = horizontal
        self.vertical = vertical
        self.space = space
        self.width = width
        self.intensity = intensity

    def apply(self, img):
        h, w, c = img.shape
        g_h = int(h / self.width)
        g_w = int(w / self.width)
        mask = np.zeros([g_h, g_w, c])

        if self.horizontal:
            mask[::self.space, :, :] = self.intensity
        if self.vertical:
            mask[:, ::self.space, :] = self.intensity

        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        img = mask.astype(np.float32) + img.astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)


class Blurring:
    def __init__(self, kernel=9, randomness=-1, seed=2022):
        self.name = "Blurring"
        if randomness == -1:
            randomness = kernel - 2
        assert 0 < randomness < kernel, "REQUIREMENT: 0 < randomness ({}) < kernel({})".format(randomness, kernel)
        self.kernel = kernel
        self.randomness = randomness
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def apply(self, img):
        k = self.kernel + self.rng.integers(-self.randomness, self.randomness)
        img = cv2.blur(img.astype(np.float32), ksize=(k, k))
        return img.astype(np.uint8)


class NeedsMoreJPG:
    def __init__(self, percentage, randomness, seed=2022):
        self.name = "NeedsMoreJPG"
        self.percentage = percentage
        self.randomness = randomness
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def apply(self, img):
        h, w = img.shape[:2]
        p = self.percentage + self.rng.integers(-self.randomness, self.randomness)
        img = cv2.resize(img, (int(w * p / 100), int(h * p / 100)), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        return img


class SaltNPepper:
    def __init__(self, max_delta, grain_size, seed=2022):
        self.name = "SaltNPepper"
        self.max_delta = max_delta
        self.grain_size = grain_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def apply(self, img):
        h, w, c = img.shape
        snp_h = max(4, int(h / self.grain_size))
        snp_w = max(4, int(w / self.grain_size))
        snp = self.rng.integers(-self.max_delta, self.max_delta, size=[snp_h, snp_w, c])
        snp = cv2.resize(snp, (w, h), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.int32) + snp
        return np.clip(img, 0, 255).astype(np.uint8)


def apply_noise(img):
    noise = SaltNPepper(
        max_delta=np.random.choice([4, 8, 16]),
        grain_size=np.random.choice([1, 2, 4, 8, 16])
    )
    return noise.apply(img)


def apply_blur(img):
    noise = Blurring(kernel=7)
    return noise.apply(img)


def apply_channel_shift(img):
    noise = ChannelShift(intensity=np.random.choice([4, 8, 16]))
    return noise.apply(img)


def apply_brightness(img):
    intensity = np.random.choice([4, 8, 16])
    p_or_m = np.random.choice([-1, 1])
    img = p_or_m * intensity + img.astype(np.int32)
    return img.astype(np.uint8)


def apply_channel_swap(img):
    swap_mode = np.random.choice([0, 1, 2, 3])
    if swap_mode == 0:
        return img[:, :, [1, 0, 2]]
    elif swap_mode == 1:
        return img[:, :, [1, 2, 0]]
    elif swap_mode == 2:
        return img[:, :, [2, 1, 0]]
    elif swap_mode == 3:
        return img[:, :, [0, 2, 1]]


def apply_horizontal_flip(img):
    img = cv2.flip(img, 1)
    return img


def apply_vertical_flip(img):
    img = cv2.flip(img, 0)
    return img


def apply_crop(img):
    height, width, ch = img.shape
    prz_zoom = 0.20
    w_random = int(width * prz_zoom)
    h_random = int(height * prz_zoom)
    if w_random > 0:
        x1_img = np.random.randint(w_random)
        x2_img = width - np.random.randint(w_random)
    else:
        x1_img = 0
        x2_img = width

    if h_random > 0:
        y1_img = np.random.randint(h_random)
        y2_img = height - np.random.randint(h_random)
    else:
        y1_img = 0
        y2_img = height

    img = img[y1_img:y2_img, x1_img:x2_img, :]
    return img


def apply_rotation_90(img):
    angle = np.random.choice([0, 90, 180, 270])
    if angle == 270:
        img = np.transpose(img, (1, 0, 2))
        img = cv2.flip(img, 0)
    elif angle == 180:
        img = cv2.flip(img, -1)
    elif angle == 90:
        img = np.transpose(img, (1, 0, 2))
        img = cv2.flip(img, 1)
    elif angle == 0:
        pass

    return img


def apply_tiny_rotation(img):
    img = img.astype(np.float64)
    rand_angle = np.random.randint(20) - 10
    img = rotate(img, angle=rand_angle, mode="reflect")
    return img.astype(np.uint8)


def apply_cut_out(img):
    height, width, ch = img.shape
    prz_zoom = 0.25
    w_random = np.random.randint(int(width * prz_zoom))
    h_random = np.random.randint(int(height * prz_zoom))
    x1_img = np.random.randint(width - w_random)
    y1_img = np.random.randint(height - h_random)

    mask = np.ones((h_random, w_random, 3))
    mask[:, :, 0] = np.random.randint(0, 255)
    mask[:, :, 1] = np.random.randint(0, 255)
    mask[:, :, 2] = np.random.randint(0, 255)

    img[y1_img:y1_img+h_random, x1_img:x1_img+w_random, :] = mask
    return img


class Augmentations:
    def __init__(self,
                flip=0, 
                rotation=0, 
                tiny_rotation=0, 
                noise=0, 
                color_shift=0, 
                brightness=0,
                blur=0, 
                crop=0, 
                channel_swap=0, 
                cut_out=0
                ):
        
        self.opt = {
            "horizontal_flip": {
                "active": flip,
                "method": apply_horizontal_flip
            },
            "vertical_flip": {
                "active": flip,
                "method": apply_vertical_flip
            },
            "noise": {
                "active": noise,
                "method": apply_noise
            },
            "color_shift": {
                "active": color_shift,
                "method": apply_channel_shift
            },
            "brighness": {
                "active": brightness,
                "method": apply_brightness
            },
            "blur": {
                "active": blur,
                "method": apply_blur
            },
            "crop": {
                "active": crop,
                "method": apply_crop
            },
            "rotation": {
                "active": rotation,
                "method": apply_rotation_90
            },
            "tiny_rotation": {
                "active": tiny_rotation,
                "method": apply_tiny_rotation
            },
            "channel_swap": {
                "active": channel_swap,
                "method": apply_channel_swap
            },
            "cut_out": {
                "active": cut_out,
                "method": apply_cut_out
            }
        }
        for k in self.opt:
            assert 0 <= self.opt[k]["active"] <= 1, "Agumentations should be in the interval [0, 1]"
        self.unique_options = [k for k in self.opt]

    def apply(self, img):
        # Augmentation (randomized)
        height, width, _ = img.shape
        new_height = np.max([height, 32])
        new_width = np.max([width, 32])
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        random.shuffle(self.unique_options)
        for k in self.unique_options:
            if self.opt[k]["active"] == 0:
                continue
            if self.opt[k]["active"] * 100 >= np.random.randint(100):
                img = self.opt[k]["method"](img)

        return img
    
    def to_dict(self):
        return {k: self.opt[k]["active"] for k in self.opt}
