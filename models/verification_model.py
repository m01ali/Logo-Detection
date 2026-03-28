import torch
from PIL import Image
import os
import getpass
from transformers import BitsAndBytesConfig, pipeline, AutoProcessor, LlavaForConditionalGeneration, PaliGemmaForConditionalGeneration


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

class LLaVAModel:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})
        
    def run_example(self, image, prompt=None):
        if prompt:
            prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        else:
            # prompt = "USER: <image>\nIs this most likely an image of a single logo only? Just give either Yes or No as the answer.\nASSISTANT:"
            prompt =  "USER: <image>\nIs this most likely an image of a single logo only? Just answer Yes or No.\nASSISTANT:"
        
        output = self.pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 128})[0]["generated_text"]
        
        result = str.lower(str.strip(str.strip(str.split(output, "ASSISTANT:")[1]), '.'))

        # print(prompt)

        # print(result)

        if  "yes" in result:
            return True
               
        return False
    
    def verify_images(self, images):
        results = [image for image in images if self.run_example(image)]
        return results

class PaligemmaModel:
    def __init__(self, model_id="google/paligemma-3b-mix-224"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, quantization_config=quantization_config).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        
    def run_example(self, image, prompt=None):
        if prompt:
            prompt = f"<image>{prompt}<bos>" 
        if not prompt:
            prompt = "<image>Does this image represent a plain icon, plain text, branded logo, or something else?<bos>"
        
        # print(prompt)
        model_inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.model.dtype)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**model_inputs, max_new_tokens=128, do_sample=False, early_stopping=False, num_beams=3)
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
        
        # print(decoded)
        return decoded
        
    def verify_image(self, image, prompt=None):
            result = self.run_example(image, prompt)
            if 'logo' in str.lower(result):
                return True
            return False
    
    def verify_images(self, images):
        results = [image for image in images if self.run_example(image)]
        return results
    
def main():
    os.environ["HF_TOKEN"] = getpass.getpass()
    model = LLaVAModel()
    logo = Image.open("ARY_Digital_Logo.png")
    
    print(model.run_example(logo))
    
if __name__ == "__main__":
    main()