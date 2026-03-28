from models.description_model import LLaMAModel
from tqdm import tqdm
from Levenshtein import ratio, distance
from itertools import combinations
import numpy as np
from enum import Enum

global_logo_prompts  = ["Is this an image of a well-known logo? Include Yes or No in your response at the start",
                    "Is this the logo of a globally recognized brand? Include Yes or No in your response at the start",
                    "Does this logo contain both design and text that represent a well-known brand? Include Yes or No in your response at the start",
                    "Is this the logo of a popular company or product? Include Yes or No in your response at the start",
                    "Is this a professional logo used by a globally established organization? Include Yes or No in your response at the start",
                    "Does this logo combine text and design elements in a way that represents a well-known brand? Include Yes or No in your response at the start",
                    "Is this logo associated with a highly recognizable global brand? Include Yes or No in your response at the start",
                    "Is this an image of a widely known corporate logo? Include Yes or No in your response at the start",
                    "Is this logo linked to a prominent multinational company? Include Yes or No in your response at the start",
                    "Is this a famous brand logo? Include Yes or No in your response at the start",
                    ]

logo_prompts = ["Is this an image of a single logo? Include Yes or No in your response at the start",
                  "Does this image represent a distinct company logo? Include Yes or No in your response at the start",
                  "Is this the logo of a brand or company or product, etc.? Include Yes or No in your response at the start",
                  "Does this image contain unique brand-specific shapes, colors, or text elements that would indicate a logo instead of a general symbol? Definitely include Yes or No in your response at the start!",
                  "Is this a logo design representing a single brand or product or company, etc.? Include Yes or No in your response at the start",
                  "Is this an image of a single professional logo? Include Yes or No in your response at the start",
                  "Is this potentially a logo of any company or product or brand, etc.? Include Yes or No in your response at the start",
                  "Does this image represent a single graphic design of a potential logo? Include Yes or No in your response at the start",
                  "Is there distinctive text placement or stylized fonts typical of a logo design? Include Yes or No in your response at the start",
                  "Would this image likely be used on a storefront, product packaging, or marketing material as a brand logo? Definitely include Yes or No in your response at the start!",
                  "Does this image contain artistic or stylized elements that suggest it is a logo? Include Yes or No in your response at the start",
                  "Could this image be used to represent a brand in marketing or advertising materials? Definitely include Yes or No in your response at the start!"]

case_prompts = {
    # 'logo': "Is this a logo of a brand or company or product, etc.? Include Yes or No at start of your response",
    # 'brand': "Is this a logo of a well-known brand? Include Yes or No at start of your response",
    'stylized': "Does this image contain any stylized text typical of logos? Include Yes or No at start of your response",
    'multiple_logos': "Is this an image showing more than one logo? Include Yes or No at start of your response",
    'multiple_logos': "Yes or No: Does this image show multiple brand logos?",
    'multiple_logos': "Yes or No: Are there more than one company logos visible in this image?",
    'multiple_logos': "Yes or No: Does the image feature multiple company logos?",
    # 'icon': "Does this image look like generic or abstract icon (menu, settings, etc.) rather than a logo that is part of a recognizable brand? Include Yes or No at start of your response",
    'poster': "Is this possibly an image of a movie poster? Include Yes or No at start of your response",
    'design': "Is this an image with design elements typical of logos but no text? Include Yes or No at start of your response",
}

class LLMResult(Enum):
    LLM = "Forward to LLM"
    API = "Forward to API"
    EXCLUDE = "EXCLUDE"

def process_image(llama_model, image):
    #Run model on global prompts for a given image
    global_logo_df = llama_model.run_examples(global_logo_prompts, image, repetitions=1, num_beams=1, temperature=0.1)
    global_logo_prob = global_logo_df['yes'].sum() / global_logo_df['include'].sum()
    print("Global Prob", global_logo_prob)
    if global_logo_prob >= 0.8:
        names = [llama_model.run_example("What is the full brand name of this logo? Only mention the brand name and nothing else. No extra words!", 
                                 image, num_beams=1, temperature=1.2) for _ in range(5)]
        names = [str.lower(name) for name in names]
        similarities = [ratio(a, b) for a, b in combinations(names, 2)]
        avg_sim = np.mean(similarities)
        if avg_sim >= 0.8:
            print("Model knows this logo very well, forward to LLM")
            return LLMResult.LLM

    #Run model on basic logo prompts for a given image
    logo_df = llama_model.run_examples(logo_prompts, image, num_beams=1, temperature=0.1)
    logo_prob = logo_df['yes'].sum() / logo_df['include'].sum()
    print("Logo Prob", logo_prob)
    
    #Run model on case prompts for a given image
    case_df = llama_model.run_examples(case_prompts.values(), image, num_beams=1, temperature=0.1)
    case_df['prompt_type'] = list(case_prompts.keys())
    multiple_logos_prob = case_df.loc[case_df.prompt_type=='multiple_logos', 'yes'].sum() /  case_df.loc[case_df.prompt_type=='multiple_logos', 'include'].sum()
    print("Multiple Logos Prob:", multiple_logos_prob)
    case_df.set_index('prompt_type', inplace=True)


    if multiple_logos_prob >= 0.67:
        print("Multiple logos so exclude")
        return LLMResult.EXCLUDE
    #If potential logo and not a poster, send to API
    if logo_prob >= 0.5 and not case_df.loc['poster', 'yes']:
        print("This is a potential logo. Forward to API")
        return LLMResult.API
    elif case_df.loc['poster', 'yes']:
        print("This is a poster so exclude")
        return LLMResult.EXCLUDE
    #If has stylized text or design elements typical of logos, send to API
    elif case_df.loc[['stylized', 'design'], 'yes'].any():
        print("Difficult Case. Forward to API")
        return LLMResult.API
    #Exclude if none of the above
    else:
        print("Exclude")
        return LLMResult.EXCLUDE