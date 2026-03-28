import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForZeroShotImageClassification, BitsAndBytesConfig
from PIL import Image, ImageDraw, ImageFont
from itertools import cycle
import numpy as np
import os
import pandas as pd
from scipy.stats import entropy
from math import log
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pathlib import Path
from io import BytesIO
import hashlib
import uuid

def compute_image_hash(logo_image):
    """Compute a unique hash for the image to be used as a logo_id."""
    try:
        buffer = BytesIO()
        logo_image.convert("RGB").save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        return hashlib.md5(image_bytes).hexdigest()
    except Exception as e:
        # Fallback to a random UUID if hashing fails
        return str(uuid.uuid4())

positive_prompts = ["This is an image of a single branded logo",
                    "This image represents a clear and distinct company logo",
                    "This is the logo of a well-known brand",
                    "This is a standalone corporate logo without extra elements",
                    "This is a minimalistic logo design representing a single brand",
                    "This is an image of a single professional logo with no distractions",
                    "This is a high-quality logo of a famous company or product",
                    "This image represents a single graphic design of a company logo",
                    "This is an isolated image of a logo used for branding purposes",
                    "This is the official logo of a brand displayed clearly",
                    "This is a focused image of one distinct logo with no additional logos or designs",
                    "This image showcases a single brand's logo without overlapping or extra elements",
                    ]

negative_prompts = ["This is not an image of a single branded logo",
                    "This is not an image of a single corporate logo",
                    "This is a random graphic or abstract image, not a logo",
                    "This is a photo of something other than a logo",
                    "This image contains multiple objects and not a single logo",
                    "This is not a clean or distinct design that represents a logo",
                    "This image shows random text or icons, not a logo",
                    "This photo represents a general illustration, not a company's branding",
                    "This is a photo or artwork unrelated to logos or branding",
                    "This image contains clutter or multiple elements, not a standalone logo",
                    "This image contains multiple logos or brand symbols, not a single distinct logo",
                    "This is a cluttered image with overlapping logos or designs from different brands",
                    ]

positive_hard_prompts = ["This is an image of a well-known logo",
                        "This is a logo of a globally recognized brand",
                        "This logo contains both design and text that represent a well-known brand",
                        "This is the logo of a popular company or product",
                        "This is a professional logo used by a globally established organization",
                        "This is a well-designed logo with a combination of text and design elements",
                        "This logo represents a highly recognizable global brand",
                        "This is an image of a widely known corporate logo",
                        "This logo is associated with a prominent multinational company",
                        "This is a famous brand logo with both visual and textual elements"]

negative_hard_prompts = ["This is a logo with only design and no text",
                        "This logo does not contain any recognizable brand name or text",
                        "This is not a logo of a globally known brand or company",
                        "This is a logo of a foreign or non-English brand that may not be widely recognized",
                        "This is an abstract design that does not represent a specific brand or company",
                        "This image lacks clear branding or association with a well-known logo",
                        "This is not a recognizable or globally known brand logo",
                        "This logo appears to belong to a small or local business, not a major brand",
                        "This is a simple graphic or design without any connection to a famous brand",
                        "This logo lacks identifiable characteristics of a globally established brand"]

# Top-k Logit Sum/Difference
def top_k_logit_sum_difference(df, k):
    pos_df = df[df['category'] == "POS"]
    neg_df = df[df['category'] == "NEG"]

    pos_sum = pos_df.head(k)['logits'].sum()
    neg_sum = neg_df.head(k)['logits'].sum()
    diff = pos_sum - neg_sum
    return pos_sum, neg_sum, diff

def rank_weighted_scores(df):
    pos_score = df[df['category'] == "POS"]['logits'].dot(df[df['category'] == "POS"]['Weight'])
    neg_score = df[df['category'] == "NEG"]['logits'].dot(df[df['category'] == "NEG"]['Weight'])
    return pos_score, neg_score, pos_score/neg_score

# Rank Position Analysis
def rank_position_analysis(df):
    pos_df = df[df['category'] == "POS"]
    neg_df = df[df['category'] == "NEG"]
    avg_pos_rank = pos_df['Rank'].mean()
    avg_neg_rank = neg_df['Rank'].mean()
    return avg_pos_rank, avg_neg_rank

# Positive to Negative Rank Ratio
def positive_to_negative_rank_ratio(df, k):
    top_k = df.head(k)
    pos_count = len(top_k[top_k['category'] == "POS"])
    neg_count = len(top_k[top_k['category'] == "NEG"])
    ratio = pos_count / (neg_count + 1e-6)  # Avoid division by zero
    return ratio

def get_ranking_metrics_one(df, top_k):
  top_k_sum_diff = top_k_logit_sum_difference(df, top_k)[-1]
  rank_weighted_score = rank_weighted_scores(df)[-1]
  pos_rank, neg_rank = rank_position_analysis(df)
  pos_to_neg_rank = neg_rank/(pos_rank + 1e-6)
  pos_to_neg_ratio = positive_to_negative_rank_ratio(df, top_k)
  return {
      "logit_sum_diff": top_k_sum_diff,
      "rank_weighted_score": rank_weighted_score,
      "avg_pos_to_neg_rank": pos_to_neg_rank,
      "count_pos_to_neg_rank": pos_to_neg_ratio
  }
  
def cluster_hard_cases(results_df, metric_cols=['score', 'logit_sum_diff', 'rank_weighted_score', 'avg_pos_to_neg_rank', 'count_pos_to_neg_rank']):
    metric_cols_z = [col + '_z' for col in metric_cols]
    for col in metric_cols:
        results_df[col+'_z'] = (results_df[col] - results_df[col].mean()) / results_df[col].std()

    pca = PCA(n_components=1)
    pca_vals = pca.fit_transform(results_df[metric_cols_z]).squeeze()
    results_df['pca_score'] = pca_vals

    kmeans = KMeans(n_clusters=3, random_state=42, )  # 3 clusters: logo, non-logo, ambiguous
    results_df['cluster'] = kmeans.fit_predict(results_df[['pca_score']])
    sorted_clusters = kmeans.cluster_centers_.mean(axis=1).argsort()

    return results_df, sorted_clusters
  
  
class CLIPModel:
    def __init__(self, model_id="openai/clip-vit-large-patch14"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def run_examples(self, images, prompts):
        inputs = self.processor(images=images, text=prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs

    def process_images(self, images, scores, positive_prompts=positive_prompts, negative_prompts=negative_prompts):
        pos_outputs = self.run_examples(images, positive_prompts)
        neg_outputs = self.run_examples(images, negative_prompts)
        pos_logits = pos_outputs.logits_per_image.cpu()
        neg_logits = neg_outputs.logits_per_image.cpu()

        combined_logits = np.concatenate([pos_logits.numpy(), neg_logits.numpy()], axis=1)
        combined_scores = torch.softmax(torch.tensor(combined_logits), dim=-1).numpy()
        combined_prompts = positive_prompts + negative_prompts
        combined_prompts_category = ['POS'] * len(positive_prompts) + ['NEG'] * len(negative_prompts)

        results = []
        for i, (image, score) in enumerate(zip(images, scores)):
          df = df = pd.DataFrame({'logits': combined_logits[i],
                          'scores': combined_scores[i],
                          'prompt': combined_prompts,
                          'category': combined_prompts_category
                          })
          df = df.sort_values(by='logits', ascending=False)
          df['Rank'] = np.arange(1, len(df) + 1)
          df['Weight'] = 1 / df['Rank']

          metrics = get_ranking_metrics_one(df, top_k=6)
          results.append({"image_path": i,
                          "score": score,
                          # "std": stds[i],
                          # 'label': labels[i],
                          **metrics})
        results_df = pd.DataFrame(results)
        return results_df
    

def detect_brands_clip(logo_inputs, clip_model, clip_threshold=0.80):
 
    logo_ids, images = zip(*logo_inputs.items())

    top_brands_1000 = pd.read_csv("../Data/fortune1000_2024.csv")
    top_brands_2000 = pd.read_csv("../Data/Top2000CompaniesGlobally.csv")
    top_brands = pd.concat([top_brands_1000[['Company']], top_brands_2000['Company']])
    top_brands.drop_duplicates('Company', inplace=True)

    brand_names = [name + " logo" for name in top_brands["Company"].to_list()]
    brand_names += ["Other", "Not a logo"]

    clip_output = clip_model.run_examples(list(images), brand_names)
    logits = clip_output.logits_per_image.softmax(dim=-1)
    max_probs, max_indices = logits.max(dim=-1)

    clip_results = {}   # {logo_id: {brand, source, prob}}
    scraper_inputs = [] # List of (logo_id, image) for logos that need scraping
    
    for logo_id, prob, idx, image in zip(logo_ids, max_probs, max_indices, images):
        if prob.item() >= clip_threshold:
            clip_results[logo_id] = {"brand_name": brand_names[idx], "source": "CLIP", "prob": prob.item()}
        else:
            scraper_inputs.append({'id': logo_id, 'image': image})
    return clip_results, scraper_inputs


if __name__ == "__main__": 
    clip_model = CLIPModel()
    logo_dir = Path("C:\\Users\\Admin\\Desktop\\milestone 2\\feedback\\feedback\\logos")
    logo_paths = list(logo_dir.glob('*.png'))

    logo_inputs = {}
    logo_to_paths ={}
    for path in logo_paths:
        try:
            img = Image.open(path)
            logo_id = compute_image_hash(img)
            logo_inputs[logo_id] = img
            logo_to_paths[logo_id] = path
        except Exception as e:
            print(f"Failed to load image {path}: {str(e)}")
        
    clip_results, scraper_inputs = detect_brands_clip(logo_inputs, clip_model, clip_threshold=0.80)
    for id, result in clip_results.items():
        print(logo_to_paths[id])
        print(result)
        print("------------")