import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, MllamaForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from PIL import Image, ImageDraw, ImageFont
from itertools import cycle
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path
import getpass
from qwen_vl_utils import process_vision_info
# from utils.prompts import QWEN_TRANSCRIPT_LOGO_PROMPT, QWEN_TRANSCRIPT_LOGO_SYSTEM_PROMPT

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)


def fixed_get_imports(filename) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports


colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

class GemmaModel:
    def __init__(self, model_id="google/gemma-2-2b"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quantization_config).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        
    def run_example(self, text, prompt=None):
        if not prompt:
            prompt = f"{text}\nThe name of the logo given in the description (not OCR) is:\n"
            
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**input_ids, max_new_tokens=32)
        outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logo_name = outputs[len(prompt):].split('\n')[0].strip()

        return logo_name

class FlorenceModel:
    def __init__(self, model_id="microsoft/Florence-2-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_id = model_id
        
        # with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=self.torch_dtype, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()
        
    def run_example(self, task_prompt: str, text_input:str = None, image: Image =None):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            # early_stopping=False,
            do_sample=False,
            num_beams=3)
        
    
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text,
                                                      task=task_prompt,
                                                      image_size=(image.width, image.height))
        
        # print(generated_ids)
        # print(generated_text)
        # print(inputs)
        
        return parsed_answer
    
    
    def describe_image(self, image: Image):
        prompt = "<MORE_DETAILED_CAPTION>"
        return self.run_example(prompt, image=image)[prompt]
    
    def extract_text(self, image: Image):
        prompt = "<OCR_WITH_REGION>"
        results = self.run_example(prompt, image=image)[prompt]
        labels = results['labels']
        labels = [str.strip(str.strip(label, "</s>")) for label in labels]
        text = " ".join(labels)
        return text
    
    def generate_description(self, image: Image):
        ocr = self.extract_text(image)
        desc = self.describe_image(image)
        result = "OCR: " + ocr + "\n" + "Description: " + desc
        return result

# class LLaMAModel:
#     def __init__(self, model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model_id = model_id
#         self.model = MllamaForConditionalGeneration.from_pretrained(
#             model_id,
#             quantization_config=quantization_config,
#             # torch_dtype=torch.bfloat16,
#             device_map="auto",
#         ).eval()
#         self.processor = AutoProcessor.from_pretrained(model_id)

#     def run_example(self, task_prompt: str, image: Image):
#         messages = [
#             {"role": "user", "content": [
#                 {"type": "image"},
#                 {"type": "text", "text": task_prompt}
#             ]}
#         ]
#         input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
#         inputs = self.processor(
#             image,
#             input_text,
#             add_special_tokens=False,
#             return_tensors="pt"
#         ).to(self.model.device, self.model.dtype)

#         output = self.model.generate(**inputs, max_new_tokens=25)
#         generated_text = self.processor.decode(output[0], skip_special_tokens=True)
#         logo_name = generated_text.strip().split('assistant')[-1].strip().strip('.')
#         return logo_name

#     def get_logo_name(self, image: Image):
#         prompt = "What is the full brand name of this logo? Only give the brand name (no special characters or extra words)"
#         return self.run_example(prompt, image=image)

class LLaMAModel:
    def __init__(self, model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            # torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def run_example(self, task_prompt: str, image: Image, **kwargs):
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": task_prompt}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            images=image,
            text=input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device, self.model.dtype)

        output = self.model.generate(**inputs, max_new_tokens=25, **kwargs)
        generated_text = self.processor.decode(output[0], skip_special_tokens=True)
        logo_name = generated_text.strip().split('assistant')[-1].strip().strip('.')
        return logo_name

    def run_examples(self, prompts, image, repetitions=1, **kwargs):        
        results = [{'prompt': prompt,
                    'answer': self.run_example(prompt, image, **kwargs)} 
                   for prompt in prompts for _ in range(repetitions)]
        results_df = pd.DataFrame(results)
        results_df['yes'] = results_df['answer'].str.lower().str.contains('yes')
        results_df['no'] = results_df['answer'].str.lower().str.contains('no')
        results_df['include'] = (results_df['yes'] | results_df['no'])
        return results_df
        
    def get_logo_name(self, image: Image):
        prompt = "What is the full brand name of this logo? Only give the brand name (no special characters or extra words)"
        return self.run_example(prompt, image=image)


class Qwen2_5VLModel:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto", 
            quantization_config=quantization_config).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True, padding_side="left", max_pixels = 128 * 28 * 28)

    def run_example(self, task_prompt: str, image: Image, **kwargs):
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": task_prompt}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            images=image,
            text=input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device, self.model.dtype)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128, **kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        output = output_text[0]
        return output_text

    def run_examples(self, prompts, images, **kwargs):        
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {
                            "type": "text", 
                            "text": prompt
                        },
                    ],
                } 
            ]
            for prompt in prompts
        ]

        #Batched Inference
        texts = [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in messages
        ]

        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device, self.model.dtype)

        generated_ids = self.model.generate(**inputs, max_new_tokens=256, **kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
    
    def chunked_run_examples(self, prompts, images, batch_size=28, **kwargs):
        results = []
        for i in tqdm(range(0, len(prompts), batch_size)):
            prompt_batch = prompts[i:i+batch_size]
            image_batch = images[i:i+batch_size]
            output = self.run_examples(prompt_batch, image_batch, **kwargs)
            results.extend(output)
        return results

        
    def get_logo_name(self, image: Image):
        prompt = "What is the full brand name of this logo? Only give the brand name (no special characters or extra words)"
        return self.run_example(prompt, image=image)

class Qwen2_5TextModel:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype="auto",
                        device_map="auto",
                        quantization_config=quantization_config,
                    ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    def run_example(self, task_prompt: str, system_prompt: str='You are a helpful assistant.', **kwargs):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
                            **model_inputs,
                            max_new_tokens=768,
                            do_sample=True,
                            temperature=1.2,
                            # temperature=0.2,
                        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


def main():
    os.environ["HF_TOKEN"] = getpass.getpass()
    # model = FlorenceModel()
    # gemma = GemmaModel()
    # llama = LLaMAModel()
    qwen = Qwen2_5VLModel(model_id="Qwen/Qwen2.5-VL-3B-Instruct")

    # logo = Image.open("ARY_Digital_Logo.png")
    # logo = Image.open("C:\\Users\\Admin\Desktop\\milestone 2\\feedback\\feedback\\logos\\9.png")
    # logo_dir = Path(f"C:\\Users\\Admin\Desktop\\milestone 2\\feedback\\feedback\\logos")
    logo_dir = Path("D:\milestone 2\logo_for_database\enel")
    # logo_paths = list(logo_dir.glob('*.png'))
    logo_paths = [
        Path("D:\milestone 2\logo_for_database\enel\\1.png"),
        Path("D:\milestone 2\logo_for_database\coca-cola\\1.png"),
        Path("D:\milestone 2\logo_for_database\coca-cola\\2.png"),
        Path("D:\milestone 2\logo_for_database\coca-cola\\3.png"),
        Path("D:\milestone 2\logo_for_database\disney\\1.png"),
        Path("D:\milestone 2\logo_for_database\dropbox\\1.png"),
        Path("D:\milestone 2\logo_for_database\MSC\\1.png"),
        Path("D:\milestone 2\logo_for_database\mapei\\1.jpeg"),
        Path("D:\milestone 2\logo_for_database\mapei\\2.jpg"),
        Path("D:\milestone 2\logo_for_database\ENI\\2.png"),
        Path("D:\milestone 2\logo_for_database\ENI\\3.png"),
        Path("D:\milestone 2\logo_for_database\lotto\\1.png"),
    ]
    logos = [Image.open(path).convert('RGB') for path in logo_paths]

    # print(len(logos))

    
    # desc = model.generate_description(logo)
    # print(desc)
    # logo_name = gemma.run_example(desc)
    # print(logo_name)
    # logo_name = [llama.run_example("What is the full brand name of this logo? Only give the brand name (no special characters or extra words)",
    #                               logo) for logo in logos]

    # prompt =  """What is the FULL brand name of this logo? Just give the FULL brand name, nothing else. 
    #             Please consider potential false positives such as generic stylized text (in movie posters for example) and only output a brand name if your confidence level is sufficiently high. 
    #             If you do not know, are unsure or the image is not a logo, just output: UNKNOWN"""

    prompt = """Extract the text given in this image. If no text is present, return an empty string. Do not include any special characters. Do NOT hallucinate!"""
    
    prompts = [prompt for _ in logos]

    logo_names = qwen.chunked_run_examples(prompts, logos, batch_size=16, temperature=0.2)
    print(logo_names)

    # model = Qwen2_5TextModel()
    # transcript = """ The great tennis on Sky is here to help you win Watch all the great tennis on Sky! When you have a sore throat, sign your days. You can try Neobosilina acts against the symptoms of sore throat with an answer suitable to all your needs sore throat the first symptoms sore throat strong sore pain with coughs. Neoborocillina, from your throat. In the grilled merluzzo grilled frost feel what counts quality so much care and authentic flavor frost 100% the natural choice. When the flu symptoms slow you down, you can try Nurofen Influencer in cold weather. Combine the anti-inflammatory decongestant action for a quick relief from the various symptoms of influenza and cold In Monifarma we want the best in life as in investments without compromises we offer you a digital platform and a dedicated consultant monifarma invest together you know that a theft on three happens from the entrance door for this very sure has developed the new alarm with intelligent maximum security lock if an intruder tries to force the lock, the alarm sends a signal to the Verishore operating station, which immediately warns the surveillance services. In addition, with a simple gesture, you open remotely to anyone you want and in case of emergency we will be there to open your door to help for an even faster intervention because with very sure your security starts from the door call now or go to berichur.it barbie the fashion collection all the barbie glamour in a collection of clothes and exclusive accessories first out and in chante d'evening at only 3 euros and 99 first release and in Chanted Evening at only 3.99 euros. I'm Orogel, I'm ready to welcome you with vegetables with a unique taste that warms the heart. Ready to make you smell the scents of the countryside? Even in the city! To be the touch of taste to your recipes. Orogel, not only ready, ready to everything! Orogel, not only ready, ready to everything! It is one of the most prestigious tournaments in the world. It's the spectacle of great volleyball. It's the regular season of Superliga. And it's a fundamental monster block. Every week on IceBog, a big match with the stars of the masculine is a pressure that is rising spectacular Italian championship series every week on Rai Sport Guinness 6 nations Scotland and Italy Saturday 1st February at 3pm on Rai 2 Rai 2. Thank you."""
    # transcript = transcript.strip()

    # result = model.run_example(
    #     QWEN_TRANSCRIPT_LOGO_PROMPT(transcript),
    #     QWEN_TRANSCRIPT_LOGO_SYSTEM_PROMPT
    # )
    # print(result)
    
if __name__ == "__main__":
    main()