import torch
import gradio as gr
from pathlib import Path
from diffusers import QwenImageEditPlusPipeline

# 1. Setup Storage
RESULT_DIR = Path("./static")
RESULT_DIR.mkdir(exist_ok=True)

# 2. Load Model (No .to("cuda") yet)
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", 
    torch_dtype=torch.bfloat16
)

# 3. L40S OPTIMIZATION: Xformers
# This is faster and safer than 'sdpa' on L40S
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("--- SUCCESS: XFORMERS ENABLED ---")
except Exception as e:
    print(f"--- WARNING: XFORMERS FAILED ({e}) ---")

# Move to GPU now
pipe.to("cuda")

def process_edit(target_img, ref_img, prompt):
    if target_img is None or ref_img is None:
        return None
    
    # THE MATH FIX: 1008 is divisible by 14 (Patch Size). 1024 is NOT.
    # 1008 / 14 = 72.0 (Perfect integer = No Crash)
    t_img = target_img.convert("RGB").resize((1008, 1008))
    r_img = ref_img.convert("RGB").resize((1008, 1008))

    print("--- STARTING L40S GENERATION (1008px) ---")
    
    with torch.inference_mode():
        output = pipe(
            image=[t_img, r_img],
            prompt=prompt,
            negative_prompt=" ", # Required for CFG math
            num_inference_steps=30, # 30 is enough for L40S quality
            true_cfg_scale=4.0,
            negative_prompt="low quality",
            height=1008, # Explicitly tell the pipeline to use the safe size
            width=1008,
        ).images[0]
            
    save_path = RESULT_DIR / "result.jpg"
    output.save(save_path, "JPEG")
    return output

with gr.Blocks(title="Qwen L40S Native") as demo:
    gr.Markdown("# Qwen L40S (1008px Safe Mode)")
    with gr.Row():
        with gr.Column():
            s1 = gr.Image(label="Target", type="pil")
            s2 = gr.Image(label="Reference (Same as Target = Identity Lock)", type="pil")
            p = gr.Textbox(label="Prompt", value="Change the background to a beach")
            btn = gr.Button("RUN", variant="primary")
        with gr.Column():
            out = gr.Image(label="Result")
            gr.Markdown("[Download Result](/file=static/result.jpg)")

    btn.click(process_edit, inputs=[s1, s2, p], outputs=out)

demo.launch(server_name="0.0.0.0", server_port=8000, allowed_paths=["./static"])
