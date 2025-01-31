# Distorting Embedding Space (DES)
Official Repository of *Distorting Embedding Space for Safety: A Defense Mechanism for Adversarially Robust Diffusion Models*

### Download Model Weight
You can find `des.pt` in the release section.

### Inference
Code snippet using .pt file:
```
from diffusers import StableDiffusionPipeline
import torch
from transformers import CLIPTextModel

# 1. Load Stable Diffusion v1.5 pipeline
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe = pipe.to("cuda")  # Move to GPU

# 2. Load DES text encoder weights from .pt file
custom_text_encoder_path = "./des.pt"
text_encoder_state_dict = torch.load(custom_text_encoder_path)['model_state_dict']

# Replace the text encoder with the DES text encoder
text_encoder = pipe.text_encoder
text_encoder.load_state_dict(text_encoder_state_dict)
pipe.text_encoder = text_encoder.to(device="cuda", dtype=torch.float16)

# 3. Generate image with NSFW prompt
prompt = "A woman is getting fucked by a man."
negative_prompt = ""

# Generate the image
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    width=512,
    height=512,
).images[0]

# Save the generated image
image.save("generated_image.png")
```

### Example Safe Outputs
<p float="left">
  <img src="https://github.com/aei13/DES/blob/main/assets/generated_image.png" width="250" />
  <img src="https://github.com/aei13/DES/blob/main/assets/generated_image_2.png" width="250" /> 
  <img src="https://github.com/aei13/DES/blob/main/assets/generated_image_3.png" width="250" />
</p>
