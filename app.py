import torch
import gradio as gr
import os
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from pathlib import Path

# Create a folder to store the static results
RESULT_DIR = Path("./static")
RESULT_DIR.mkdir(exist_ok=True)

# Load the model
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", 
    torch_dtype=torch.bfloat16
).to("cuda")

def process_edit(target_img, ref_img, prompt):
    if target_img is None or ref_img is None:
        return None
    
    # TRIMMING/RESIZE LOGIC: Force Ref to match Target exactly
    # We use 1024x1024 as the standard for Qwen 2509 stability
    t_img = target_img.convert("RGB").resize((1024, 1024))
    r_img = ref_img.convert("RGB").resize((1024, 1024))

    # CFG MATH FIX: You MUST pass a string to negative_prompt if true_cfg_scale > 1
    inputs = {
        "image": [t_img, r_img],
        "prompt": prompt,
        "negative_prompt": " ", # This stops the 'calculation shit' from crashing
        "num_inference_steps": 40,
        "true_cfg_scale": 4.0,
        "height": 1024,
        "width": 1024,
    }
    
    with torch.inference_mode():
        output = pipe(**inputs).images[0]
    
    # Save locally so it's fetchable via URL
    save_path = RESULT_DIR / "result.jpg"
    output.save(save_path, "JPEG")
    
    return output

with gr.Blocks(title="Qwen Fixed") as demo:
    gr.Markdown("# Qwen Image Edit - Identity & Style Locked")
    with gr.Row():
        with gr.Column():
            slot1 = gr.Image(label="SLOT 1: TARGET", type="pil")
            slot2 = gr.Image(label="SLOT 2: REFERENCE", type="pil")
            prompt = gr.Textbox(label="Instruction", value="Use image 2 to edit image 1.")
            btn = gr.Button("RUN", variant="primary")
        with gr.Column():
            out = gr.Image(label="RESULT")
            url_display = gr.Markdown("Direct Link: [result.jpg](/file=static/result.jpg)")

    btn.click(fn=process_edit, inputs=[slot1, slot2, prompt], outputs=out)

# ALLOWED_PATHS lets you access the /static/result.jpg via the browser
demo.launch(
    server_name="0.0.0.0", 
    server_port=8000, 
    debug=True, 
    allowed_paths=["./static"]
)
