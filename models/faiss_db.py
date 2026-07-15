import os
import json
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import faiss
import glob
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
import pandas as pd
import torch
# from rapidfuzz import process, fuzz
# import easyocr
import re


def _select_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _extract_image_embeddings(clip_model: CLIPModel, clip_inputs: dict) -> torch.Tensor:
    """Return image embeddings as a tensor across transformers API variants."""
    pixel_values = clip_inputs.get("pixel_values")
    if pixel_values is None:
        raise RuntimeError("CLIP inputs missing 'pixel_values'")

    with torch.no_grad():
        try:
            embeddings = clip_model.get_image_features(pixel_values=pixel_values)
        except Exception:
            vision_outputs = clip_model.vision_model(pixel_values=pixel_values)
            if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
                pooled = vision_outputs.pooler_output
            elif hasattr(vision_outputs, "last_hidden_state") and vision_outputs.last_hidden_state is not None:
                pooled = vision_outputs.last_hidden_state[:, 0, :]
            else:
                raise RuntimeError("Unable to extract pooled visual features from CLIP vision model")

            if hasattr(clip_model, "visual_projection") and clip_model.visual_projection is not None:
                embeddings = clip_model.visual_projection(pooled)
            else:
                embeddings = pooled

    if isinstance(embeddings, torch.Tensor):
        return embeddings
    if hasattr(embeddings, "image_embeds") and isinstance(embeddings.image_embeds, torch.Tensor):
        return embeddings.image_embeds
    if hasattr(embeddings, "pooler_output") and isinstance(embeddings.pooler_output, torch.Tensor):
        return embeddings.pooler_output

    raise RuntimeError(f"Unexpected embedding type: {type(embeddings)}")

def apply_augmentation(image: Image.Image, augmenter) -> Image.Image:
    """
    Apply an Albumentations augmenter to a PIL image.

    Args:
        image: PIL Image
        augmenter: Albumentations Compose or similar callable

    Returns:
        Augmented PIL Image
    """
    img_np = np.array(image)
    augmented = augmenter(image=img_np)['image']
    return Image.fromarray(augmented)



class LogoDatabase:
    """
    A production-ready logo similarity search system with batch processing.
    
    Features:
    - Batch embedding generation with configurable size
    - Direct PIL image input/output
    - Optimized metadata storage
    - Efficient duplicate checking
    - GPU acceleration support
    """
    
    def __init__(self, 
                 index_path: str = "logo_index.faiss",
                 metadata_path: str = "metadata.json",
                 device: str = _select_torch_device(),
                 batch_size: int = 8):
        """
        Initialize the logo database.
        
        Args:
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load brand metadata
            device: Compute device for embedding model
            batch_size: Number of images to process simultaneously
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.device = device
        self.batch_size = batch_size
        
        # Initialize DINOv2 model
        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large', use_fast=True)
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-large').to(device)

        # Initialize CLIP model and processor (using the large variant)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        
        # Initialize FAISS index and metadata
        self.index = None
        self.metadata = []
        self._init_index()

    def _init_index(self):
        """Initialize or load existing FAISS index"""

        # combined_dim = 1024 + 768
        combined_dim=768
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(combined_dim)  # DINOv2-large dimension
            self.metadata = []
        
        if self.device == 'cuda' and hasattr(faiss, 'StandardGpuResources'):
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception as e:
                print(f"GPU FAISS unavailable ({e}); using CPU FAISS index.")

    def _embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Generate DINOv2 embeddings for a batch of images.
        
        Args:
            images: List of PIL Image objects (RGB format)
            
        Returns:
            Normalized embedding vectors (batch_size x 1024)
        """
        try:
            clip_inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
            combined_embeddings = _extract_image_embeddings(self.clip_model, clip_inputs)
            combined_embeddings = combined_embeddings.cpu().numpy()
            norms = np.linalg.norm(combined_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-12, norms)
            combined_embeddings_normalized = combined_embeddings / norms
            return combined_embeddings_normalized
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            return np.array([])
        

    def add_logos(self, 
                 images: List[Image.Image], 
                 brand_names: List[str],
                 duplicate_threshold: float = 0.95):
        """
        Add new logos to the database with batch processing.
        
        Args:
            images: Iterable of PIL Image objects (RGB format)
            brand_names: Corresponding brand names
            duplicate_threshold: Similarity threshold for duplicate detection
        """
        images = list(images)
        if len(images) != len(brand_names):
            raise ValueError("Number of images and brand names must match")
            
        total_added = 0
        for i in tqdm(range(0, len(images), self.batch_size)):
            batch_images = images[i:i+self.batch_size]
            batch_brands = brand_names[i:i+self.batch_size]
            
            # Generate embeddings for batch
            batch_embeddings = self._embed_batch(batch_images)
            if batch_embeddings.size == 0:
                continue
                
            # Check for duplicates in existing database
            # if self.index.ntotal > 0:
            #     # Find nearest existing neighbors
            #     distances, _ = self.index.search(batch_embeddings, 1)
            #     similarities = 1 - distances.flatten() / 4
                
            #     # Mask for new entries
            #     mask = similarities < duplicate_threshold
            #     batch_embeddings = batch_embeddings[mask]
            #     batch_brands = [b for b, m in zip(batch_brands, mask) if m]
                
            # Add valid entries to index and metadata
            if batch_embeddings.size > 0:
                self.index.add(batch_embeddings.astype('float32'))
                self.metadata.extend(batch_brands)
                total_added += len(batch_brands)
                
        print(f"Added {total_added} new logos (skipped {len(images)-total_added} duplicates)")

    def search_logo(self, 
                   query_image: Image.Image, 
                   threshold: float = 0.85,
                   k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar logos in the database.
        
        Args:
            query_image: PIL Image object (RGB format)
            threshold: Minimum similarity score (0-1)
            k: Number of neighbors to retrieve
            
        Returns:
            List of (brand_name, similarity_score) tuples
        """
        # Process single image as batch of 1
        batch_embed = self._embed_batch([query_image])
        if batch_embed.size == 0:
            return []
            
        query_embed = batch_embed[0].astype('float32')
        distances, indices = self.index.search(np.expand_dims(query_embed, 0), k)
        
        results = []
        for i, dist in zip(indices[0], distances[0]):
            similarity = 1 - dist / 4  # Convert L2 to cosine-like similarity
            if similarity >= threshold and i < len(self.metadata):
                results.append({'index': i, 
                                'brand_name': self.metadata[i], 
                                'similarity': similarity
                               })
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)

    def save(self):
        """Persist the index and metadata to disk"""
        faiss.write_index(faiss.index_gpu_to_cpu(self.index), self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)

    def __len__(self):
        return len(self.metadata)

    @property
    def total_logos(self):
        return len(self.metadata)
    
    
class LogoDatabaseNew:
    """
    A production-ready logo similarity search system with batch processing.
    
    Features:
    - Batch embedding generation with configurable size
    - Direct PIL image input/output
    - Optimized metadata storage
    - Efficient duplicate checking
    - GPU acceleration support
    """
    
    def __init__(self, 
                 index_path: str = "logo_index.faiss",
                 metadata_path: str = "metadata.json",
                 device: str = _select_torch_device(),
                 batch_size: int = 8):
        """
        Initialize the logo database.
        
        Args:
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load brand metadata
            device: Compute device for embedding model
            batch_size: Number of images to process simultaneously
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.device = device
        self.batch_size = batch_size
        
        # Initialize DINOv2 model
        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-large').to(device)

        # Initialize CLIP model and processor (using the large variant)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        
        # Initialize FAISS index and metadata
        self.index = None
        self.metadata = []
        self._init_index()

    def _init_index(self):
        """Initialize or load existing FAISS index"""

        # combined_dim = 1024 + 768
        combined_dim=768
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(combined_dim)  # DINOv2-large dimension
            self.metadata = []
        
        if self.device == 'cuda' and hasattr(faiss, 'StandardGpuResources'):
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception as e:
                print(f"GPU FAISS unavailable ({e}); using CPU FAISS index.")

    def _embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Generate DINOv2 embeddings for a batch of images.
        
        Args:
            images: List of PIL Image objects (RGB format)
            
        Returns:
            Normalized embedding vectors (batch_size x 1024)
        """
        try:
            clip_inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
            combined_embeddings = _extract_image_embeddings(self.clip_model, clip_inputs)
            combined_embeddings = combined_embeddings.cpu().numpy()
            norms = np.linalg.norm(combined_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-12, norms)
            combined_embeddings_normalized = combined_embeddings / norms
            return combined_embeddings_normalized
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            return np.array([])

    def add_logos(self, 
                  images: List[Image.Image], 
                  brand_names: List[str],
                  duplicate_threshold: float = 0.95,
                  augmentations = None,
                  num_augments: int = 10):
        """
        Add new logos to the database with optional augmentations.
    
        Args:
            images: List of PIL Image objects (RGB)
            brand_names: Corresponding brand names
            duplicate_threshold: Optional threshold for deduplication (unused currently)
            augmentations: Albumentations Compose object or similar callable
            num_augments: Number of augmentations to generate per image (if augmentations are provided)
        """
        if len(images) != len(brand_names):
            raise ValueError("Number of images and brand names must match")
    
        all_images = []
        all_brands = []
    
        for img, brand in zip(images, brand_names):
            all_images.append(img)
            all_brands.append(brand)
    
            # Apply augmentations if requested to store multiple variants along with original image
            if augmentations:
                for _ in range(num_augments):
                    aug_img = apply_augmentation(img, augmentations)
                    all_images.append(aug_img)
                    all_brands.append(brand)
    
        total_added = 0
        for i in tqdm(range(0, len(all_images), self.batch_size)):
            batch_images = all_images[i:i + self.batch_size]
            batch_brands = all_brands[i:i + self.batch_size]
    
            # Generate embeddings
            batch_embeddings = self._embed_batch(batch_images)
            if batch_embeddings.size == 0:
                continue
    
            # Add to index + metadata
            self.index.add(batch_embeddings.astype('float32'))
            self.metadata.extend(batch_brands)
            total_added += len(batch_brands)
    
        print(f"Added {total_added} new logos including augmentations.")



    def search_logo(self, 
                    query_image: Image.Image, 
                    threshold: float = 0.85,
                    k: int = 5,
                    augmentations = None,
                    num_augments: int = 5):
        """
        Search for similar logos in the database using optional test-time augmentation.

        Args:
            query_image: PIL Image (RGB)
            threshold: Minimum similarity score (0–1)
            k: Number of neighbors to retrieve
            augmentations: Albumentations Compose object (optional)
            num_augments: Number of augmented versions to include (if augmentations are provided)
    
        Returns:
            List of dictionaries with keys: 'index', 'brand_name', 'similarity'
        """
        # Build list of query images (original + augmentations)
        query_images = [query_image]

        #Apply test-time augmentations
        if augmentations:
            img_np = np.array(query_image)
            for _ in range(num_augments):
                aug_np = augmentations(image=img_np)['image']
                query_images.append(Image.fromarray(aug_np))
    
        # Generate embeddings in batch
        embeddings = self._embed_batch(query_images)
        if embeddings.size == 0:
            return []
    
        # Normalize each embedding, then average them, then re-normalize the average embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        avg_embedding = np.mean(normalized, axis=0)
        avg_embedding /= np.linalg.norm(avg_embedding)
    
        # FAISS search
        distances, indices = self.index.search(np.expand_dims(avg_embedding.astype('float32'), 0), k)
        
        results = []
        for i, dist in zip(indices[0], distances[0]):
            similarity = 1 - dist / 4
            if similarity >= threshold and i < len(self.metadata):
                results.append({
                    'index': i,
                    'brand_name': self.metadata[i],
                    'similarity': similarity
                })
    
        return sorted(results, key=lambda x: x['similarity'], reverse=True)

    def search_logos(self,
                     images: List[Image.Image],
                     threshold: float = 0.85,
                     k: int = 5,
                     batch_size: int = 64,
                     augmentations = None,
                     num_augments: int = 5):
        
        """
        Perform batched similarity search for a list of logos with optional test-time augmentation.
    
        Args:
            images: List of PIL Image objects (RGB)
            threshold: Minimum similarity score (0 to 1)
            k: Number of nearest neighbors to retrieve per image
            batch_size: Number of image variants to embed per batch
            augmentations: Albumentations Compose object (optional)
            num_augments: Number of augmentations per image if augmentations provided
    
        Returns:
            List of results (one per original image). Each result is a list of dictionaries.
        """
        
        all_results = []
    
        for img in images:
            # Test-time augmentation Prepare augmented variants for this particular image
            variants = [img]
            if augmentations:
                img_np = np.array(img)
                for _ in range(num_augments):
                    aug_np = augmentations(image=img_np)['image']
                    variants.append(Image.fromarray(aug_np))
    
            # Embed all variants in batch with original
            emb = self._embed_batch(variants)
            if emb.size == 0:
                all_results.append([])
                continue
    
            # Normalize and average, then re-normalize average embedding
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            avg_emb = np.mean(emb, axis=0)
            avg_emb /= np.linalg.norm(avg_emb)
    
            # Search FAISS
            distances, indices = self.index.search(np.expand_dims(avg_emb.astype('float32'), 0), k)
            
            image_results = []
            for i, dist in zip(indices[0], distances[0]):
                similarity = 1 - dist / 4
                if similarity >= threshold and i < len(self.metadata):
                    image_results.append({
                        'index': i,
                        'brand_name': self.metadata[i],
                        'similarity': similarity
                    })
            
            all_results.append(sorted(image_results, key=lambda x: x['similarity'], reverse=True))
    
        return all_results



    def save(self):
        """Persist the index and metadata to disk"""
        index = self.index
        if self.device=='cuda':
            index = faiss.index_gpu_to_cpu(index)
            
        faiss.write_index(index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)

    def __len__(self):
        return len(self.metadata)

    @property
    def total_logos(self):
        return len(self.metadata)
    



class LogoDatabaseSigLIP2(LogoDatabaseNew):
    """
    Drop-in variant of LogoDatabaseNew that uses SigLIP2 embeddings (1152-dim)
    instead of CLIP (768-dim).

    Requires a SEPARATE FAISS index file — the two models produce incompatible
    embedding spaces.  All add/search/save logic is inherited from LogoDatabaseNew;
    only the embedding model and FAISS initialisation are overridden.
    """

    EMBED_DIM = 1152
    MODEL_ID = "google/siglip2-so400m-patch14-384"

    def __init__(
        self,
        index_path: str = "logo_index_siglip2.faiss",
        metadata_path: str = "metadata_siglip2.json",
        device: str = _select_torch_device(),
        batch_size: int = 8,
    ):
        # Intentionally bypass LogoDatabaseNew.__init__ so CLIP is never loaded.
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.device = device
        self.batch_size = batch_size

        self._siglip_proc = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        self._siglip_model = AutoModel.from_pretrained(self.MODEL_ID).to(self.device)
        self._siglip_model.eval()

        self.index = None
        self.metadata = []
        self._init_index()

    # ------------------------------------------------------------------
    # Override: initialise FAISS with the correct 1152-dim embedding space
    # ------------------------------------------------------------------
    def _init_index(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.EMBED_DIM)
            self.metadata = []

        if self.device == "cuda" and hasattr(faiss, 'StandardGpuResources'):
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception as e:
                print(f"GPU FAISS unavailable ({e}); using CPU FAISS index.")

    # ------------------------------------------------------------------
    # Override: embed with SigLIP2 instead of CLIP
    # ------------------------------------------------------------------
    def _embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        try:
            rgb = [img.convert("RGB") for img in images]
            inputs = self._siglip_proc(images=rgb, return_tensors="pt").to(self.device)
            with torch.no_grad():
                vision_out = self._siglip_model.vision_model(
                    pixel_values=inputs["pixel_values"]
                )
                if hasattr(vision_out, "pooler_output") and vision_out.pooler_output is not None:
                    pooled = vision_out.pooler_output
                else:
                    pooled = vision_out.last_hidden_state[:, 0, :]

                if (
                    hasattr(self._siglip_model, "visual_projection")
                    and self._siglip_model.visual_projection is not None
                ):
                    emb = self._siglip_model.visual_projection(pooled)
                else:
                    emb = pooled

            arr = emb.cpu().numpy()
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            return arr / np.where(norms == 0, 1e-12, norms)
        except Exception as e:
            print(f"SigLIP2 embed error: {e}")
            return np.array([])


if __name__ == '__main__':
    index_path = "D:\\milestone 2\\faiss_db\\logo_index.faiss"
    metadata_path = "D:\\milestone 2\\faiss_db\\metadata.json"

    db = LogoDatabase(index_path, metadata_path, device='cpu', batch_size=32)


    logos_path = Path("D:\\milestone 2\\faiss_db\\logo_for_database\\logo_for_database")

    logo = Image.open(logos_path/'frosta'/'1.png').convert('RGB')

    result = db.search_logo(logo,
                            threshold=0.85,
                            k=5)
    
    print(result[0])