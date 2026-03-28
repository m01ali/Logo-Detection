import gradio as gr
# from detector_model import LogoDetector
from models.description_model import FlorenceModel, GemmaModel, LLaMAModel
from models.verification_model import LLaVAModel, PaligemmaModel
import os


desc_model = FlorenceModel()
gemma_model = None
verify_model_A = LLaVAModel()
paligemma_model = None
llama_model = None


def process_image(hf_key, image, task, llava_prompt, pg_prompt):
  if hf_key:
      os.environ["HF_TOKEN"] = hf_key
  else:
      return "Please set a valid Huggingface API Key"
  
  try:
    global paligemma_model, gemma_model, llama_model
    if not paligemma_model:
        paligemma_model = PaligemmaModel()

    if not gemma_model:
        gemma_model = GemmaModel()
    
    if not llama_model:
        llama_model = LLaMAModel()

    if task=="Verify":
        result_A = verify_model_A.run_example(image, llava_prompt)
        result_B = paligemma_model.verify_image(image, pg_prompt)
        result =  f"LLaVAModel's Output: {result_A}\nPaligemmaModel's Output: {result_B}"
    else:
        desc = desc_model.generate_description(image)
        result_A = gemma_model.run_example(desc)
        result_B = paligemma_model.run_example(image, prompt="What is the full brand name of this logo?")
        result_C = llama_model.get_logo_name(image)
        result = f"FlorenceGemma's Output: {result_A}\nPaligemma's Output: {result_B}\nLLaMA 3.2's Output: {result_C}"
  except Exception as e:
      result = str(e)

  return result
  
  

DESCRIPTION = "# Automated Logo Detection - Phase 2"

css = """
  #output {
    height: 500px;
    overflow: auto;
    border: 1px solid #ccc;
  }
"""

VERIFY_PG_PROMPT = "Does this image represent a plain icon, plain text, branded logo, or something else?"
VERIFY_LLAVA_PROMPT = "Is this most likely an image of a single logo only? Just answer Yes or No."


with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tab(label="Logo Verification and Brand Recognition"):
        # with gr.Row():
              
        hf_key = gr.Textbox(label="Huggingface_API_KEY", placeholder="Huggingface API KEY", type="password")

        input_img = gr.Image(label="Input Frame", type="pil")

        task = gr.Radio(["Verify", "Describe"], label="Select Task")

        llava_prompt = gr.Textbox(value=VERIFY_LLAVA_PROMPT, label="Prompt for the LLaVA verification model only.")
        pg_prompt = gr.Textbox(value=VERIFY_PG_PROMPT, label="Prompt for the Paligemma verification model only.")

        submit_btn = gr.Button(value="Submit")

        output_text = gr.Textbox(value="", label="Model Output")
    
        # output_img = gr.Image(label="Output Image")
        # output_logos = gr.Gallery(columns=1, label="Cropped Logos", preview=True, show_label=True)
      
    with gr.Accordion("Instructions"):
        gr.Markdown(
            """
            - Upload an image which is either a logo or something else.
            - Select the task you want to perform (verification or description).
            - Click "Submit" to execute the desired task.
            - The output will be shown on the text box at the bottom.
            """)

    submit_btn.click(process_image, [hf_key, input_img, task, llava_prompt, pg_prompt], [output_text])

demo.launch(debug=True)