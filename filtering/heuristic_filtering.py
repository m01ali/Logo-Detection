import cv2
import numpy as np
from PIL import Image, ImageStat
from typing import List, Callable, Optional
from pipelines.brand_recognition_chain import LogoImage


# class LogoImageFilter:
#     def __init__(self, id: str, crop: Image.Image, frame_area: Optional[int] = None):
#         self.id = id
#         self.crop = crop
#         self.frame_area = frame_area

# Individual heuristic functions

def area_aspect_filter(
    crop: Image.Image,
    min_area: int = 500,
    max_area_ratio: float = 0.5,
    max_aspect_ratio: float = 4.0,
    frame_area: Optional[int] = None
) -> bool:
    """
    Reject crops with implausible area or extreme aspect ratios.
    """
    w, h = crop.size
    area = w * h
    if area < min_area:
        return False
    if frame_area and (area / frame_area) > max_area_ratio:
        return False
    aspect = max(w / (h + 1e-6), h / (w + 1e-6))
    if aspect > max_aspect_ratio:
        return False
    return True


def texture_filter(
    crop: Image.Image,
    min_std: float = 12.0
) -> bool:
    """
    Require a minimum grayscale standard deviation (texture) to reject uniform areas.
    """
    gray = crop.convert('L')
    std = ImageStat.Stat(gray).stddev[0]
    return std >= min_std


def edge_density_filter(
    crop: Image.Image,
    min_edge_frac: float = 0.05
) -> bool:
    """
    Use Canny edge detection to ensure sufficient edge density.
    """
    arr = np.array(crop.convert('L'))
    edges = cv2.Canny(arr, 50, 150)
    edge_frac = np.mean(edges > 0)
    return edge_frac >= min_edge_frac


def color_variance_filter(
    crop: Image.Image,
    min_color_std: float = 10.0
) -> bool:
    """
    Ensure sufficient color variance (stddev across RGB channels).
    """
    arr = np.array(crop)
    # Compute stddev across pixels per channel
    stds = np.std(arr.reshape(-1, 3), axis=0)
    return np.mean(stds) >= min_color_std


class FilterPipeline:
    """
    Composes multiple heuristics into a single filter pipeline.

    Example:
        pipeline = FilterPipeline([
            area_aspect_filter,
            texture_filter,
            edge_density_filter,
            color_variance_filter,
        ])
        pruned = pipeline.apply(logo_images)
    """
    def __init__(self, filters: List[Callable[..., bool]]):
        self.filters = filters

    def apply(
        self,
        logo_images: List[LogoImage]
    ) -> List[LogoImage]:
        """
        Apply all filters in sequence to each LogoImage. Keeps those passing all.
        """
        pruned: List[LogoImage] = []
        for li in logo_images:
            passed = True
            for fn in self.filters:
                # Pass frame_area if function accepts it
                try:
                    if fn.__code__.co_argcount >= 2 and 'frame_area' in fn.__code__.co_varnames:
                        if not fn(li.image, frame_area=li.metadata.get('frame_area', None)):
                            passed = False
                            break
                    else:
                        if not fn(li.image):
                            passed = False
                            break
                except Exception as e:
                    print(e)
                    passed = False
                    break
            if passed:
                pruned.append(li)
        return pruned