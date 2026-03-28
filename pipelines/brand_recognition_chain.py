from typing import List, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from PIL import Image
import uuid
from filtering.clip_filtering import CLIPModel, positive_prompts, negative_prompts, detect_brands_clip
import pandas as pd
import re
from utils.fuzzy_matching import combine_easyocr_text_ordered, get_ocr_image, robust_fuzzy_match, robust_fuzzy_match_batch, smart_score
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from typing import List, Tuple, Optional, Dict, Any
import albumentations as A
from models.description_model import Qwen2_5VLModel
from collections import Counter

try:
    # If RapidFuzz is already in your env, we’ll use it for the soft boost.
    from rapidfuzz import fuzz, process
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False


# --- 1. LogoImage Data Model ---

@dataclass
class LogoImage:
    id: str
    image: Image.Image
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(image: Image.Image, metadata: Dict[str, Any] = None) -> 'LogoImage':
        return LogoImage(
            id=str(uuid.uuid4()),
            image=image,
            metadata=metadata or {},
        )

# --- 2. Abstract Brand Recognition Technique ---

class BrandRecognitionTechnique(ABC):
    @abstractmethod
    def predict(self, logo_images: List[LogoImage]) -> Dict[str, str]:
        """
        Given a list of LogoImage, returns a dict mapping image IDs to brand names (or 'UNKNOWN').
        """
        pass

# --- 3. Brand Recognizer Chain Engine ---

# class BrandRecognizer:
#     def __init__(self, techniques: List[BrandRecognitionTechnique]):
#         self.techniques = techniques

#     def recognize(self, logo_images: List[LogoImage]) -> Dict[str, Dict]:
#         remaining_images = {logo.id: logo for logo in logo_images}
#         final_results = {}

#         for technique in self.techniques:
#             if not remaining_images:
#                 break

#             predictions = technique.predict(list(remaining_images.values()))

#             for image_id, brand in predictions.items():
#                 if brand != 'UNKNOWN':
#                     logo = remaining_images[image_id]
#                     final_results[image_id] = {
#                         'brand': brand,
#                         'image': logo.image,
#                         'metadata': logo.metadata,
#                     }

#             # Only keep images still UNKNOWN
#             remaining_images = {
#                 img_id: logo for img_id, logo in remaining_images.items()
#                 if img_id not in final_results
#             }

#         # Mark any leftovers explicitly as UNKNOWN
#         for image_id, logo in remaining_images.items():
#             final_results[image_id] = {
#                 'brand': 'UNKNOWN',
#                 'image': logo.image,
#                 'metadata': logo.metadata,
#             }

#         return final_results


class BrandRecognizer:
    def __init__(self, techniques: List[BrandRecognitionTechnique]):
        self.techniques = techniques

    def recognize(self, logo_images: List[LogoImage]) -> Dict[str, Dict[str, Any]]:
        remaining_images = {logo.id: logo for logo in logo_images}
        final_results = {}

        for technique in self.techniques:
            if not remaining_images:
                break

            predictions = technique.predict(list(remaining_images.values()))

            processed = 0
            for image_id, prediction in predictions.items():
                if isinstance(prediction, str):
                    # Simple string prediction (just brand name)
                    brand = prediction
                    extra_info = {}
                elif isinstance(prediction, dict):
                    # Rich prediction with additional info
                    brand = prediction.get('brand', 'UNKNOWN')
                    extra_info = {k: v for k, v in prediction.items() if k != 'brand'}
                else:
                    raise ValueError("Prediction must be either str or dict.")

                if brand != 'UNKNOWN':
                    logo = remaining_images[image_id]
                    final_results[image_id] = {
                        'brand': brand,
                        'technique': technique.__class__.__name__,
                        'image': logo.image,
                        'metadata': logo.metadata,
                        **extra_info  # Add any extra fields like 'score', 'actual_text'
                    }
                    processed+= 1
            
            total_images = len(remaining_images)
            failed = total_images - processed

            print(f"[{technique.__class__.__name__}] Processed {processed} / {total_images} images successfully. {failed} remain unrecognized.")

                
            # Only keep logos that are still UNKNOWN
            remaining_images = {
                img_id: logo for img_id, logo in remaining_images.items()
                if img_id not in final_results
            }

        # Mark leftover images explicitly as UNKNOWN
        for image_id, logo in remaining_images.items():
            final_results[image_id] = {
                'brand': 'UNKNOWN',
                'technique': None,
                'image': logo.image,
                'metadata': logo.metadata
            }

        return final_results

# --- 4. CLIP-Based Brand Recognition Technique ---


# class ClipTechnique(BrandRecognitionTechnique):
#     def __init__(self, clip_model, clip_threshold: float = 0.8):
#         """
#         clip_model: CLIP model instance with run_examples method
#         clip_threshold: Probability threshold for accepting predictions
#         """
#         self.clip_model = clip_model
#         self.clip_threshold = clip_threshold
#         self.brand_names = self._load_brand_prompts()

#         self.llm = Qwen2_5VLModel(model_id="Qwen/Qwen2.5-VL-3B-Instruct")
#         self.llm_prompt = """Extract the text given in this image. If no text is present, return an empty string. Do not include any special characters. Do NOT hallucinate!"""


#     def _load_brand_prompts(self) -> List[str]:
#         # Load brands once during init
#         # top_brands_1000 = pd.read_csv("../Data/fortune1000_2024.csv")
#         # top_brands_2000 = pd.read_csv("../Data/Top2000CompaniesGlobally.csv")
#         top_brands = pd.read_csv("D:\milestone 2\Data\Deduplicated_Companies.csv")
#         top_brands.rename(columns={'Brand': 'Company'}, inplace=True)
#         # top_brands = pd.concat([top_brands_1000[['Company']], top_brands_2000['Company']])
#         top_brands.drop_duplicates('Company', inplace=True)

#         brand_names = [name + " logo" for name in top_brands["Company"].to_list()]
#         brand_names += ["Other", "Not a logo"]
#         return brand_names

#     def predict(self, logo_images: List[LogoImage]) -> Dict[str, str]:
#         results = {}

#         if not logo_images:
#             return results
        
#         images = [li.image for li in logo_images]
#         ids = [li.id for li in logo_images]

#         llm_prompts = [self.llm_prompt for _ in images]
#         ocr_texts = self.llm.chunked_run_examples(llm_prompts, images, batch_size=32, temperature=0.2)
#         print(ocr_texts)

#         # Run CLIP model
#         clip_output = self.clip_model.run_examples(images, self.brand_names)
#         logits = clip_output.logits_per_image.softmax(dim=-1)
#         max_probs, max_indices = logits.max(dim=-1)

#         for logo_id, prob, idx in zip(ids, max_probs, max_indices):
#             if prob.item() >= self.clip_threshold:
#                 results[logo_id] = self.brand_names[idx]
#             else:
#                 results[logo_id] = 'UNKNOWN'

#         return results

class ClipTechnique(BrandRecognitionTechnique):
    def __init__(
        self,
        clip_model,
        clip_threshold: float = 0.8,
        fuzzy_threshold: int = 90
    ):
        """
        clip_model: CLIP model instance with run_examples method
        clip_threshold: Probability threshold for accepting CLIP predictions
        fuzzy_threshold: Similarity (0–100) threshold for accepting fuzzy OCR matches
        """
        self.clip_model = clip_model
        self.clip_threshold = clip_threshold
        self.fuzzy_threshold = fuzzy_threshold

        # --- LOAD BASE COMPANY NAMES (no suffix) ---
        base_df = pd.read_csv(r"D:\milestone 2\Data\Deduplicated_Companies.csv")
        base_df.rename(columns={'Brand':'Company'}, inplace=True)
        base_df.drop_duplicates('Company', inplace=True)
        self.base_brand_names = base_df["Company"].tolist()

        # --- CLIP PROMPTS: add " logo" + catch-alls ---
        self.clip_brand_names = [
            name + " logo" for name in self.base_brand_names
        ] + ["Other", "Not a logo"]

        # --- NORMALIZED LIST FOR FUZZY MATCHING ---
        self.normalized_fuzzy_brand_names = [
            self._normalize(name) for name in self.base_brand_names
        ]

        # LLM for OCR
        self.llm = Qwen2_5VLModel(model_id="Qwen/Qwen2.5-VL-3B-Instruct")
        self.llm_prompt = (
            "Extract the text given in this image. If no text is present, return "
            "an empty string. Do not include any special characters. Do NOT hallucinate!"
        )

        # NEW prompt that asks the LLM to self-filter blurry/unreadable images
        # self.llm_prompt = """
        # You are given an image. Your job is to extract any legible text from it.
        # If you cannot confidently read any text—because it is too blurry, low resolution,
        # or otherwise unreadable, please return an EMPTY STRING!
        # Do not hallucinate any characters or words!!!
        # Do not include punctuation or special characters.
        # """

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Lowercase, strip out punctuation/special chars, collapse whitespace.
        """
        text = text.lower()
        text = re.sub(r'[\r\n\t]', ' ', text)
        text = re.sub(r'[^a-z0-9 ]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def predict(self, logo_images: List[LogoImage]) -> Dict[str, str]:
        results: Dict[str, str] = {}
        if not logo_images:
            return results

        images = [li.image for li in logo_images]
        ids = [li.id for li in logo_images]

        # 1) OCR
        prompts = [self.llm_prompt] * len(images)
        ocr_texts = self.llm.chunked_run_examples(
            prompts, images, batch_size=32, temperature=0.2
        )
        norm_ocr = [self._normalize(txt) for txt in ocr_texts]

        # 2) Fuzzy-match texts ≥ 3 chars against base names
        long_idxs = [i for i, txt in enumerate(norm_ocr) if len(txt) >= 4]
        clip_candidates = set(range(len(images)))

        if long_idxs:
            queries = [norm_ocr[i] for i in long_idxs]
            print(queries)
            # batch cdist: shape (len(queries), len(base_brand_names))
            sim_mat = process.cdist(
                queries,
                self.normalized_fuzzy_brand_names,
                scorer=fuzz.WRatio
            )

            for q_i, sims in enumerate(sim_mat):
                orig_i = long_idxs[q_i]
                best_sim = sims.max()
                best_j = int(sims.argmax())
                if best_sim >= self.fuzzy_threshold:
                    # assign the plain company name
                    assigned_brand = self.base_brand_names[best_j]
                    results[ids[orig_i]] = {
                        'brand': assigned_brand,
                        'source': "Top_2K_Brands",
                    }
                    clip_candidates.remove(orig_i)

        # 3) CLIP on everything not resolved by fuzzy
        if clip_candidates:
            c_imgs = [images[i] for i in clip_candidates]
            c_ids = [ids[i] for i in clip_candidates]

            clip_out = self.clip_model.run_examples(
                c_imgs, self.clip_brand_names
            )
            logits = clip_out.logits_per_image.softmax(dim=-1)
            max_p, max_idx = logits.max(dim=-1)

            for lid, p, j in zip(c_ids, max_p, max_idx):
                if p.item() >= self.clip_threshold:
                    results[lid] = { 
                        "brand": self.clip_brand_names[j],
                        "source": "Top_2K_Brands" 
                    }
                else:
                    results[lid] = {"brand": "UNKNOWN"}

        return results


# --- 5. OCR + Fuzzy Matching Brand Recognition Technique ---

class OcrFuzzyTechnique(BrandRecognitionTechnique):
    def __init__(self, ocr_func, fuzzy_matcher, known_brands: List[str], fuzzy_threshold: float = 0.8):
        self.ocr_func = ocr_func
        self.fuzzy_matcher = fuzzy_matcher
        self.known_brands = [b.lower().strip() for b in known_brands]
        self.fuzzy_threshold = fuzzy_threshold

    def predict(self, logo_images: List[LogoImage]) -> Dict[str, str]:
        results = {}

        for logo in logo_images:
            ocr_text = self.ocr_func(logo.image)
            if not ocr_text or len(ocr_text) < 3:
                results[logo.id] = 'UNKNOWN'
                continue

            matched_brand, score = self.fuzzy_matcher(ocr_text, self.known_brands)
            # if score >= self.fuzzy_threshold:
            #     results[logo.id] = matched_brand
            # else:
            results[logo.id] = {
                'brand': matched_brand,
                'fuzzy_score': score,
                'ocr_text': ocr_text,
                'source': 'audio',
            }

        return results
    

class OcrFuzzyBatchTechnique(BrandRecognitionTechnique):
    def __init__(
        self,
        ocr_func,
        fuzzy_matcher,
        known_brands: List[str],
        fuzzy_threshold: float = 0.8
    ):
        """
        Args:
            ocr_func: Callable that takes a PIL Image and returns OCR text.
            known_brands: List of canonical brand names.
            fuzzy_threshold: Float in [0,1], converted internally to 0–100.
        """
        self.ocr_func = ocr_func
        self.fuzzy_matcher = fuzzy_matcher
        # Pre-normalize your brand list
        self.known_brands = [b.lower().strip() for b in known_brands]
        # Map to an integer threshold for RapidFuzz (0–100)
        self.base_threshold = int(fuzzy_threshold * 100)

    def predict(self, logo_images: List[LogoImage]) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}

        # 1. Extract all OCR texts in one pass
        ocr_texts = []
        for logo in logo_images:
            raw = self.ocr_func(logo.image) or ""
            ocr_texts.append(raw.strip())

        # 2. Batch fuzzy‐match them against known_brands
        matches = self.fuzzy_matcher(
            texts=ocr_texts,
            brand_list=self.known_brands,
            base_threshold=self.base_threshold
        )

        # 3. Unpack into your results dict
        for logo, ocr_txt, (brand, score) in zip(logo_images, ocr_texts, matches):
            results[logo.id] = {
                "brand":       brand,
                "fuzzy_score": score,
                "ocr_text":    ocr_txt,
                "source": "audio",
            }

        return results


# --- 5. FAISS + OCR + Fuzzy Matching Brand Recognition Technique ---
AUGMENTATIONS = A.Compose([
    A.Rotate(limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.ColorJitter(hue=0.05, saturation=0.1, p=0.4),
    A.Resize(224, 224)
])

class DatabaseFaissTechnique(BrandRecognitionTechnique):
    def __init__(self, db, ocr_reader, base_threshold: int = 75, faiss_threshold: float = 0.85, min_text_len: int = 3):
        self.db = db
        self.ocr_reader = ocr_reader
        self.base_threshold = base_threshold
        self.faiss_threshold = faiss_threshold
        self.min_text_len = min_text_len

    def _extract_clean_text(self, image: Image.Image) -> str:
        text = get_ocr_image(image).strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z\s]", "", text)
        return text

    def predict(self, logo_images: List[LogoImage]) -> Dict[str, Any]:
        results = {}

        for logo in logo_images:
            search_results = self.db.search_logo(logo.image, k=5, threshold=self.faiss_threshold)
            ocr_text = self._extract_clean_text(logo.image)

            if not search_results:
                results[logo.id] = {
                    'brand': 'UNKNOWN',
                    'ocr_text': ocr_text,
                    'matched_text': None,
                    'score': 0.0
                }
                continue

            top_result = search_results[0]
            db_text = top_result['brand_name'].lower().strip()
            db_score = top_result['similarity']

            if not ocr_text and not db_text and db_score >= 0.93:
                results[logo.id] = {
                    'brand': db_text,
                    'ocr_text': '',
                    'matched_text': db_text,
                    'score': db_score
                }
            else:
                matched_text = robust_fuzzy_match(ocr_text, db_text, self.base_threshold, self.min_text_len)
                brand = matched_text if matched_text != "UNKNOWN" else "UNKNOWN"
                results[logo.id] = {
                    'brand': brand,
                    'ocr_text': ocr_text,
                    'matched_text': db_text,
                    'score': db_score
                }

        return results


class DatabaseFaissParallelTechnique(BrandRecognitionTechnique):
    def __init__(self, db, ocr_reader, base_threshold: int = 75, faiss_threshold: float = 0.85, min_text_len: int = 3, num_workers: int = None):
        self.db = db
        self.ocr_reader = ocr_reader
        self.base_threshold = base_threshold
        self.faiss_threshold = faiss_threshold
        self.min_text_len = min_text_len
        self.num_workers = num_workers if num_workers else max(1, mp.cpu_count() - 2)

    def _extract_clean_text(self, image: Image.Image) -> str:
        text = get_ocr_image(image).strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z\\s]", "", text)
        return text

    def _process_logo(self, logo: LogoImage):
        """Process a single LogoImage into a (logo_id, result_dict)"""
        search_results = self.db.search_logo(logo.image, k=5, threshold=self.faiss_threshold)
        ocr_text = self._extract_clean_text(logo.image)

        if not search_results:
            return logo.id, {
                'brand': 'UNKNOWN',
                'ocr_text': ocr_text,
                'matched_text': None,
                'score': 0.0
            }

        top_result = search_results[0]
        db_text = top_result['brand_name'].lower().strip()
        db_score = top_result['similarity']

        if not ocr_text and not db_text and db_score >= 0.93:
            return logo.id, {
                'brand': db_text,
                'ocr_text': '',
                'matched_text': db_text,
                'score': db_score
            }
        else:
            matched_text = robust_fuzzy_match(ocr_text, db_text, self.base_threshold, self.min_text_len)
            brand = matched_text if matched_text != "UNKNOWN" else "UNKNOWN"
            return logo.id, {
                'brand': brand,
                'ocr_text': ocr_text,
                'matched_text': db_text,
                'score': db_score
            }

    def predict(self, logo_images: List[LogoImage]) -> Dict[str, Any]:
        results = {}

        if not logo_images:
            return results

        with mp.Pool(self.num_workers) as pool:
            # Partial is not needed here, function has no extra args
            logo_result_pairs = pool.map(self._process_logo, logo_images)

        # Collect results
        for logo_id, result in logo_result_pairs:
            results[logo_id] = result

        return results


# --- Refactored Technique ---
# class DatabaseFaissTechniqueBatch(BrandRecognitionTechnique):
#     def __init__(
#         self,
#         db,
#         ocr_reader,
#         base_threshold: int = 75,
#         faiss_threshold: float = 0.85,
#         min_text_len: int = 3,
#     ):
#         self.db = db
#         self.ocr_reader = ocr_reader
#         self.base_threshold = base_threshold
#         self.faiss_threshold = faiss_threshold
#         self.min_text_len = min_text_len

#     def _extract_clean_text(self, image: Image.Image) -> str:
#         text = get_ocr_image(image).strip().lower()
#         text = re.sub(r"\s+", " ", text)
#         # text = re.sub(r"[^a-z\s]", "", text)
#         return text

#     # def predict(self, logo_images: List[LogoImage]) -> Dict[str, Any]:
#     #     results = {}

#     #     # 1) Batch FAISS search on all images
#     #     images = [logo.image for logo in logo_images]
#     #     batch_search = self.db.search_logos(
#     #         images,
#     #         threshold=self.faiss_threshold,
#     #         k=5,
#     #         batch_size=self.db.batch_size,
#     #         augmentations=AUGMENTATIONS,
#     #         num_augments=3,
#     #     )
#     #     # batch_search[i] is a list of up to k matches dicts for logo i

#     #     # 2) Extract OCR texts
#     #     ocr_texts = [self._extract_clean_text(img) for img in images]

#     #     # 3) Pick top-1 db_text per image (or "" if none)
#     #     top_db_texts  = []
#     #     top_db_scores = []
#     #     for matches in batch_search:
#     #         if matches:
#     #             bm = matches[0]
#     #             top_db_texts.append(bm["brand_name"].lower().strip())
#     #             top_db_scores.append(bm["similarity"])
#     #         else:
#     #             top_db_texts.append("")
#     #             top_db_scores.append(0.0)


#     #     # print(top_db_texts, top_db_scores)

#     #     # 4) Batch‐match OCR→DB strings
#     #     matched_texts = robust_fuzzy_match_batch(
#     #         ocr_texts,
#     #         top_db_texts,
#     #         base_threshold=self.base_threshold,
#     #         min_text_len=self.min_text_len,
#     #     )

#     #     # 5) Build results
#     #     for idx, logo in enumerate(logo_images):
#     #         db_text  = top_db_texts[idx]
#     #         db_score = top_db_scores[idx]
#     #         ocr_txt  = ocr_texts[idx]
#     #         best_match = matched_texts[idx]

#     #         # If no FAISS hit at all
#     #         if not batch_search[idx]:
#     #             brand = "UNKNOWN"
#     #         elif db_text == "false_logo" and db_score >= 0.94:
#     #             brand = "UNKNOWN"
#     #         # Special case: design-only query image but high FAISS confidence
#     #         # elif (not ocr_txt or (len(ocr_txt) == 0)) and db_score >= 0.95:
#     #         elif not ocr_txt and not db_text and db_score >= 0.93:
#     #             brand = db_text
#     #             best_match = db_text
#     #         # Successful FAISS + fuzzy match
#     #         else:
#     #             brand = best_match if best_match != "UNKNOWN" else "UNKNOWN"

#     #         results[logo.id] = {
#     #             "brand":       brand,
#     #             "ocr_text":    ocr_txt,
#     #             "matched_text": best_match if best_match != "UNKNOWN" else None,
#     #             "score":       db_score
#     #         }

#     #     return results


#     def predict(self, logo_images: List[LogoImage]) -> Dict[str, Any]:
#         results = {}

#         # 1) Get top‑k FAISS matches
#         images = [logo.image for logo in logo_images]
#         batch_search = self.db.search_logos(
#             images,
#             threshold=self.faiss_threshold,
#             k=5,
#             batch_size=self.db.batch_size,
#             augmentations=AUGMENTATIONS,
#             num_augments=3,
#         )

#         # 2) OCR texts
#         ocr_texts = [self._extract_clean_text(img) for img in images]

#         # Tunables
#         fuzzy_threshold = 85   # percent
#         min_ocr_len     = 3    # require at least this many chars to try fuzzy

#         for idx, logo in enumerate(logo_images):
#             matches = batch_search[idx]
#             ocr_txt = ocr_texts[idx] or ""
#             ocr_len = len(ocr_txt)

#             # A) FALSE_LOGO guard on top‑1 match
#             if matches and matches[0]["brand_name"].lower().strip() == "false_logo" and matches[0]["similarity"] >= 0.93:
#                 results[logo.id] = {
#                     "brand":        "UNKNOWN",
#                     "ocr_text":     ocr_txt,
#                     "matched_text": None,
#                     "score":        0.0
#                 }
#                 continue

#             # B) Need OCR text to proceed
#             if ocr_len < min_ocr_len or not matches:
#                 # either too short for fuzzy or no FAISS hits
#                 results[logo.id] = {
#                     "brand":        "UNKNOWN",
#                     "ocr_text":     ocr_txt,
#                     "matched_text": None,
#                     "score":        0.0
#                 }
#                 continue

#             # C) Compute fuzzy over top‑k names
#             names        = [m["brand_name"] for m in matches]
#             # fuzzy_scores = [fuzz.WRatio(ocr_txt.lower().strip(), nm.lower().strip()) for nm in names]
#             fuzzy_scores = [smart_score(ocr_txt.lower().strip(), nm.lower().strip()) for nm in names]

#             best_i     = int(np.argmax(fuzzy_scores))
#             best_fuzzy = fuzzy_scores[best_i]
#             best_name  = names[best_i]

#             if best_fuzzy >= fuzzy_threshold:
#                 # Accept that best fuzzy candidate
#                 results[logo.id] = {
#                     "brand":        best_name,
#                     "ocr_text":     ocr_txt,
#                     "matched_text": best_name,
#                     "score":        best_fuzzy / 100.0
#                 }
#             else:
#                 # None met the threshold → UNKNOWN
#                 results[logo.id] = {
#                     "brand":        "UNKNOWN",
#                     "ocr_text":     ocr_txt,
#                     "matched_text": None,
#                     "score":        0.0
#                 }

#         return results




class DatabaseFaissTechniqueBatch(BrandRecognitionTechnique):
    def __init__(
        self,
        db,
        ocr_reader,
        base_threshold: int = 75,
        faiss_threshold: float = 0.85,
        min_text_len: int = 3,
        # --- NEW: minimal knobs (safe defaults keep current behavior) ---
        visual_strict_threshold: float = 0.97,     # accept top-1 by visuals alone
        visual_consensus_threshold: float = 0.90,  # check consensus if top-1 >= this
        consensus_topk: int = 5,                   # reuse your existing k=5
        consensus_min_count: int = 4,              # require >=4 same-name votes
        text_min_threshold: int = 80,              # lowered fuzzy threshold (was 85)
        soft_boost_enable: bool = True,            # +5 if OCR roughly matches any top-K name
        # --- NEW: toggle Qwen OCR like in ClipTechnique ---
        use_qwen_ocr: bool = False,
        qwen_batch_size: int = 32,
        qwen_temperature: float = 0.2,
        qwen_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    ):
        self.db = db
        self.ocr_reader = ocr_reader
        self.base_threshold = base_threshold
        self.faiss_threshold = faiss_threshold
        self.min_text_len = min_text_len

        # --- NEW: store knobs ---
        self.visual_strict_threshold = visual_strict_threshold
        self.visual_consensus_threshold = visual_consensus_threshold
        self.consensus_topk = consensus_topk
        self.consensus_min_count = consensus_min_count
        self.text_min_threshold = text_min_threshold
        self.soft_boost_enable = soft_boost_enable

        # --- NEW: Qwen OCR config ---
        self.use_qwen_ocr = use_qwen_ocr
        self.qwen_batch_size = qwen_batch_size
        self.qwen_temperature = qwen_temperature
        self.qwen_model_id = qwen_model_id

        if self.use_qwen_ocr:
            # Mirror your ClipTechnique setup
            self.llm = Qwen2_5VLModel(model_id=self.qwen_model_id)
            self.llm_prompt = (
                "Extract the text given in this image. If no text is present, return an empty string. "
                "Do not include any special characters. Do NOT hallucinate!"
            )

    # --- Existing EasyOCR path (kept) ---
    def _extract_clean_text(self, image: Image.Image) -> str:
        text = get_ocr_image(image).strip().lower()
        text = re.sub(r"\s+", " ", text)
        # text = re.sub(r"[^a-z\s]", "", text)
        return text

    # --- NEW: normalization to mirror ClipTechnique behavior for Qwen path ---
    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[\r\n\t]', ' ', text)
        text = re.sub(r'[^a-z0-9 ]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # --- NEW: unified OCR batch (Qwen batched or EasyOCR per-image) ---
    def _extract_ocr_batch(self, images: List[Image.Image]) -> List[str]:
        if self.use_qwen_ocr:
            # Batch call Qwen exactly like your ClipTechnique
            prompts = [self.llm_prompt] * len(images)
            try:
                ocr_texts = self.llm.chunked_run_examples(
                    prompts, images, batch_size=self.qwen_batch_size, temperature=self.qwen_temperature
                )
                return [self._normalize(txt) for txt in ocr_texts]
            except Exception:
                # Defensive fallback to EasyOCR path if Qwen fails
                return [self._extract_clean_text(img) for img in images]
        else:
            # Keep existing behavior
            return [self._extract_clean_text(img) for img in images]

    # --- NEW: simple majority/consensus helper over top-K names ---
    def _consensus_stats(self, names: List[str]) -> Tuple[Optional[str], int, float]:
        """
        Returns (best_name, count, share) for the most common name in top-K.
        """
        if not names:
            return None, 0, 0.0
        counts = Counter([n.strip().lower() for n in names if n])
        best_name, best_cnt = max(counts.items(), key=lambda kv: kv[1])
        share = best_cnt / max(1, len(names))
        return best_name, best_cnt, share

    # --- NEW: tiny soft boost for fuzzy if OCR ≈ any top-K name ---
    def _soft_boost(self, text: str, topk_names: List[str]) -> int:
        """
        +5 points if OCR text roughly matches any top-K name.
        Uses RapidFuzz token_set_ratio when available; otherwise a lenient heuristic.
        """
        if not self.soft_boost_enable or not text or not topk_names:
            return 0

        if _HAS_RAPIDFUZZ:
            for nm in topk_names:
                if fuzz.token_set_ratio(text, nm.lower().strip()) >= 70:
                    return 5
            return 0
        else:
            # Fallback heuristic: token overlap proportion ≥ 0.5
            tset = set(text.split())
            for nm in topk_names:
                nset = set(nm.lower().strip().split())
                inter = len(tset & nset)
            if inter >= max(1, 0.5 * max(len(tset), len(nset))):
                return 5
            return 0

    def predict(self, logo_images: List[LogoImage]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        # 1) Get top-k FAISS matches (unchanged)
        images = [logo.image for logo in logo_images]
        batch_search = self.db.search_logos(
            images,
            threshold=self.faiss_threshold,
            k=5,
            batch_size=self.db.batch_size,
            augmentations=AUGMENTATIONS,
            num_augments=3,
        )

        # 2) OCR texts (NOW unified & batched, Qwen or EasyOCR)
        ocr_texts = self._extract_ocr_batch(images)

        # Tunables (retain your existing names; keep behavior consistent)
        fuzzy_threshold = self.text_min_threshold
        min_ocr_len     = self.min_text_len

        for idx, logo in enumerate(logo_images):
            matches = batch_search[idx]
            ocr_txt = (ocr_texts[idx] or "").strip().lower()
            ocr_len = len(ocr_txt)

            # A) FALSE_LOGO guard on top-1 match (unchanged)
            if matches and matches[0]["brand_name"].lower().strip() == "false_logo" and matches[0]["similarity"] >= 0.93:
                results[logo.id] = {
                    "brand":        "UNKNOWN",
                    "ocr_text":     ocr_txt,
                    "matched_text": None,
                    "score":        0.0
                }
                continue

            # Short-circuit if no FAISS hits at all
            if not matches:
                results[logo.id] = {
                    "brand":        "UNKNOWN",
                    "ocr_text":     ocr_txt,
                    "matched_text": None,
                    "score":        0.0
                }
                continue

            # --- NEW: VISUAL-ONLY PATH BEFORE ANY OCR REQUIREMENTS ---

            # B1) STRICT VISUAL GATE
            top1_sim = float(matches[0]["similarity"])
            top1_name = matches[0]["brand_name"]
            if top1_sim >= self.visual_strict_threshold:
                results[logo.id] = {
                    "brand":        top1_name,
                    "ocr_text":     ocr_txt,
                    "matched_text": top1_name,
                    "score":        top1_sim,  # keep similarity as score for visual accept,
                    "source":       "database",
                }
                continue

            # B2) CONSENSUS GATE (requires already good visual top-1)
            if top1_sim >= self.visual_consensus_threshold:
                names_topk = [m["brand_name"] for m in matches[:self.consensus_topk]]
                best_name, best_cnt, share = self._consensus_stats(names_topk)
                # Conservative: also require share ≥ 0.4 to be safe
                if (best_name and 
                    best_cnt >= self.consensus_min_count and 
                    share >= 0.4 and 
                    best_name.lower().strip() == top1_name.lower().strip()): #best name is same as top1 name
                    
                    results[logo.id] = {
                        "brand":        best_name,
                        "ocr_text":     ocr_txt,
                        "matched_text": best_name,
                        "score":        top1_sim,
                        "source":       "database",
                    }
                    continue

            # --- TEXT PATH (only if we have enough OCR text) ---
            if ocr_len < min_ocr_len:
                # Not enough OCR for text path → UNKNOWN (visual gates already tried)
                results[logo.id] = {
                    "brand":        "UNKNOWN",
                    "ocr_text":     ocr_txt,
                    "matched_text": None,
                    "score":        0.0
                }
                continue

            # C) Compute fuzzy over top-k names (unchanged core; current soft boost line commented in your version)
            names = [m["brand_name"] for m in matches]
            fuzzy_scores = [smart_score(ocr_txt, nm.lower().strip()) for nm in names]
            best_i       = int(np.argmax(fuzzy_scores))
            best_fuzzy   = float(fuzzy_scores[best_i])
            best_name    = names[best_i]

            # If you want the soft boost back, uncomment:
            # boosted = best_fuzzy + self._soft_boost(ocr_txt, names)
            boosted = best_fuzzy

            if boosted >= fuzzy_threshold:
                results[logo.id] = {
                    "brand":        best_name,
                    "ocr_text":     ocr_txt,
                    "matched_text": best_name,
                    "score":        boosted / 100.0,  # keep your score convention for text path
                    "source":       "database",
                }
            else:
                results[logo.id] = {
                    "brand":        "UNKNOWN",
                    "ocr_text":     ocr_txt,
                    "matched_text": None,
                    "score":        0.0
                }

        return results