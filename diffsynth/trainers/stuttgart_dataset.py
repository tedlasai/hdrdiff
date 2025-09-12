import torch, torchvision, imageio, os, json, pandas
import imageio.v3 as iio
from PIL import Image
import os
    
from pathlib import Path
import re
import cv2
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)



class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)



class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data



class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)



class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)



class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)



class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image



class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image



class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    


class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        reader = imageio.get_reader(data)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames


def _natural_key(s: str):
    # splits into digit/non-digit chunks so "file2" < "file10"
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]

def next_paths(path: str, t: int, *, same_suffix: bool = True, include_self: bool = False):
    """
    Given a file path, return the next t paths in that directory in natural sorted order.
    
    Args:
        path: The starting file path.
        t: How many following paths to return.
        same_suffix: If True, only consider files with the same extension as `path`.
        include_self: If True, include `path` itself as the first element (then next t-1).
        
    Returns:
        List[str]: up to t subsequent paths (or t including self if include_self=True).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {path}")

    directory = p.parent
    suffix = p.suffix

    # Gather candidates
    if same_suffix:
        candidates = [f for f in directory.iterdir() if f.is_file() and f.suffix == suffix]
    else:
        candidates = [f for f in directory.iterdir() if f.is_file()]

    # Sort naturally by name
    candidates.sort(key=lambda f: _natural_key(f.name))

    # Find index of the given file
    try:
        idx = candidates.index(p)
    except ValueError:
        # If the file isn't in the filtered list (e.g., suffix mismatch), fall back to all files
        all_files = sorted([f for f in directory.iterdir() if f.is_file()],
                           key=lambda f: _natural_key(f.name))
        try:
            idx = all_files.index(p)
            candidates = all_files  # use this list going forward
        except ValueError:
            raise FileNotFoundError(f"File not found among directory listings: {path}")

    # Slice out next t entries (optionally including self)
    start = idx if include_self else idx + 1
    result = candidates[start:start + t]

    return [str(f) for f in result]


def make_exposure_brackets(hdr_paths, exposures=(-5, 0, 5)):
    """
    Given a list of HDR image paths, generate exposure-bracketed LDR images.

    Args:
        hdr_paths (list[str]): Paths to HDR images.
        exposures (tuple[int|float]): EV values to apply for exposure scaling.

    Returns:
        list[list[np.ndarray]]: For each HDR path, a list of LDR images
                                (same order as exposures).
    """
    all_brackets = []
    hdr_images = []
    for hdr_path in hdr_paths:
        # Read HDR as float32, convert BGR -> RGB
        hdr_in = cv2.imread(hdr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[:, :, ::-1]
        hdr_images.append(hdr_in)

        ldr_images = []
        for ev in exposures:
            # Scale exposure (2^EV), clip to [0,1]
            ldr = np.clip(hdr_in * (2.0 ** ev), 0.0, 1.0)
            # Convert to uint8
            ldr = (ldr * 255.0).astype(np.uint8)
            ldr_images.append(ldr)

        all_brackets.append(ldr_images)
    hdr_images = np.array(hdr_images)  # shape (N, H, W, 3)
    all_brackets = np.array(all_brackets)  # shape (N, len(exposures), H, W, 3)
    all_brackets = all_brackets.transpose(1,0,2,3,4)  # shape (len(exposures), N, H, W, 3)

    return hdr_images, all_brackets

class LoadHDRVideo(DataProcessingOperator):
    def __init__(self, num_frames=49, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        #get num_frames-1 next frames with same suffix 
        #assert that num_frames is 3n+1
        assert (self.num_frames - 1) % 3 == 0, "num_frames must be 3n+1"
        num_hdr_frames = (self.num_frames - 1) // 3
        hdr_paths = next_paths(data, num_hdr_frames, same_suffix=True, include_self=True)
 
        hdr_frames, frames = make_exposure_brackets(hdr_paths, exposures=(0,-4,4))

        frames = frames.reshape(-1, *frames.shape[2:])  # shape (num_frames, H, W, 3)
        #repeat first frame in the begining
        frames = np.concatenate([frames[0:1], frames], axis=0)

        pil_frames = []
        for frame_id in range(frames.shape[0]):
            frame = Image.fromarray(frames[frame_id])
            frame = self.frame_processor(frame)
            pil_frames.append(frame)
            
        return hdr_frames, pil_frames


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]



class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames
    


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")



class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")



class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)



class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)



class StuttgartDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None,
        repeat=1,
        main_data_operator=lambda x: x,
        special_operator_map=None,
        mode = "brackets"
    ):
        self.base_path = base_path
        self.repeat = repeat
        self.data_file_keys = ["video"]
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.mode = mode
        self.data = []
        self.cached_data = {}
        self.load_from_cache = False
        self.load_data_from_path()
        

    def load_data_from_path(self):
        files = sorted(os.listdir(self.base_path))
        files = files[:-17] #I need at least 16 HDR frames to build a video
        self.data = [
            {
                "prompt": "", 
                "video": files[i],
            }
            for i in range(len(files))
        ]

    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[(str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                        num_frames, time_division_factor, time_division_remainder,
                        frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                    )),
                (("hdr", "exr"), LoadHDRVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ]))
        ])

            

    def __getitem__(self, data_id):
        if (data_id % len(self.data)) in self.cached_data:
            data = self.cached_data[data_id % len(self.data)]
        else:
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    if self.mode == "hdr_and_brackets" and key == "video":
                       data["hdr_video"], data[key] = self.main_data_operator(data[key])
                    elif key in self.special_operator_map:
                        data[key] = self.special_operator_map[key]
                    elif key in self.data_file_keys:
                        _, data[key] = self.main_data_operator(data[key])
            self.cached_data[data_id % len(self.data)] = data
        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
    
