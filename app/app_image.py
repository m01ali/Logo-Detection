import gradio as gr
from models.detector_model import LogoDetector
from models.description_model import FlorenceModel
from models.verification_model import LLaVAModel, PaligemmaModel
import os

model = LogoDetector()
# desc_model = FlorenceModel()
verify_model = None

def process_image(hf_key, image, thresh, nms_thresh):
  if hf_key:
      os.environ["HF_TOKEN"] = hf_key
  else:
      return "Please set a valid Huggingface API Key"
  
  global verify_model
  if not verify_model:
      verify_model = PaligemmaModel()
  
  annotated_image, potential_logos = model.process_image(image, thresh, nms_thresh)

  logos = [logo for logo in potential_logos if verify_model.verify_image(logo)]

  descriptions = [verify_model.run_example(logo, "What is the full brand name of this logo?") for logo in logos]
  # logos = verify_model.verify_images(potential_logos)
  # descriptions = desc_model.describe_images(logos)
  return annotated_image, list(zip(logos, descriptions))
  
  


DESCRIPTION = "# Automated Logo Detection - Phase 1"

css = """
  #output {
    height: 500px;
    overflow: auto;
    border: 1px solid #ccc;
  }
"""


with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tab(label="Logo Detection"):
        hf_key = gr.Textbox(label="Huggingface_API_KEY", placeholder="Huggingface API KEY", type="password")
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="Input Frame", type="pil")

                submit_btn = gr.Button(value="Detect Logos")

                input_threshold = gr.Slider(
                  label="Threshold",
                  info="Larger value will detect fewer logos and vice versa.",
                  minimum=0.01,
                  maximum=0.2,
                  value=0.1,
                  step=0.01)
                
                input_nms_threshold = gr.Slider(
                  label="NMS_Threshold",
                  info="Larger value will detect fewer overlapping logos and vice versa.",
                  minimum=0.1,
                  maximum=0.9,
                  value=0.3,
                  step=0.1)

            with gr.Column():
                # output_text = gr.Textbox(label="Output Text")
                output_img = gr.Image(label="Output Image")
                output_logos = gr.Gallery(columns=1, label="Cropped Logos", preview=True, show_label=True)
        
        with gr.Accordion("Instructions"):
          gr.Markdown(
              """
              - Upload an image containing logos.
              - Select a threshold using the slider.
              - Click "Detect Logos" to detect logos in the given image.
              - The output image with bounding boxes drawn for each logo will be shown in the top right.
              - The cropped logos will be shown in the gallery below the output image.
              """)

        submit_btn.click(process_image, [hf_key, input_img, input_threshold, input_nms_threshold], [output_img, output_logos])

demo.launch(debug=False)