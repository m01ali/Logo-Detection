import gradio as gr
# from detector_model import LogoDetector
from utils.shot_detection import ShotDetector
from pipelines.pipeline import process_video, process_frame_results_merged
# from description_model import FlorenceModel
# from verification_model import LLaVAModel, PaligemmaModel
import os



def run_pipeline(hf_key, video, threshold, detector_threshold, detector_nms_threshold, use_db: bool, use_translation: bool, use_llm_db: bool, use_qwen_filter: bool):
  if hf_key:
      os.environ["HF_TOKEN"] = hf_key
  else:
      return "Please set a valid Huggingface API Key"
  
  frame_results = process_video(video, threshold, detector_threshold, detector_nms_threshold, use_db, use_translation, use_llm_db, use_qwen_filter)
  # results_df["brand_name"] = results_df["brand_name"].str.replace('\n', ' or ')
  # results_df['info'] = "Brand Name(s): " + results_df['brand_name'] + ", Start Timestamp: " + results_df['start_timestamp'] + ", End Timestamp: " + results_df['end_timestamp']

  results_df = process_frame_results_merged(frame_results)


  logos = results_df['logo'].values.tolist()
  infos = results_df['info'].values.tolist()
  # frames, frame_numbers, timestamps = detector.process_video(quantile=threshold)
  # print(len(frames), len(frame_numbers), len(timestamps))
  return list(zip(logos, infos))
  


DESCRIPTION = "# Automated Logo Detection - Phase 3"

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
                input_video = gr.Video(label="Input Video")

                submit_btn = gr.Button(value="Run Pipeline")

                input_threshold = gr.Slider(
                  label="Video Threshold (Quantile)",
                  info="Larger value will detect fewer keyframes from video and vice versa",
                  minimum=0.80,
                  maximum=0.99,
                  value=0.98,
                  step=0.01)
                
                detector_threshold = gr.Slider(
                  label="Detector Threshold",
                  info="Larger value will detect fewer logos within a keyframe vice versa",
                  minimum=0.005,
                  maximum=0.1,
                  value=0.03,
                  step=0.005)
                
                
                detector_nms_threshold = gr.Slider(
                  label="Detector NMS Threshold",
                  info="Larger value will remove more overlapping logos/regions and vice versa",
                  minimum=0.05,
                  maximum=0.4,
                  value=0.3,
                  step=0.05)
                
                
                use_db_checkbox = gr.Checkbox(label="Use Database?", value=False)

                use_llm_db = gr.Checkbox(label="Use Qwen for Database OCR?", value=False)

                use_translation_checkbox = gr.Checkbox(label="Use Engilish Translation?", value=False)

                use_qwen_filter = gr.Checkbox(label="Use Qwen for False Positive Filtering?", value=False)
                


            with gr.Column():
                # output_text = gr.Textbox(label="Output Text")
                # output_img = gr.Image(label="Output Image")
                output_frames = gr.Gallery(columns=1, label="Keyframes with timestamps", preview=True, show_label=True)
        
        with gr.Accordion("Instructions"):
          gr.Markdown(
              """
              - Upload a video containing logos.
              - Select a threshold using the slider.
              - Click "Run Pipeline" to detect logos in the given video.
              - The extracted logos with corresponding brand names, start & end timestamps will be shown in the gallery on the right
              """)

        submit_btn.click(run_pipeline, [hf_key, input_video, input_threshold, detector_threshold, detector_nms_threshold, use_db_checkbox, use_translation_checkbox, use_llm_db, use_qwen_filter], [output_frames])

if __name__ == '__main__':
   demo.launch(debug=True)