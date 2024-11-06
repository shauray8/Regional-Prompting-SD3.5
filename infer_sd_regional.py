import torch
from pipeline_sd3_regional import RegionalStableDiffusion3Pipeline, RegionalStableDiffusion3AttnProcessor2_0

if __name__ == "__main__":
    
    model_path = "stabilityai/stable-diffusion-3.5-large"
    
    use_lora = False
    use_controlnet = False

    pipeline = RegionalStableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16, token="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").to("cuda")
    
    attn_procs = {}
    for name in pipeline.transformer.attn_processors.keys():
        if 'transformer_blocks' in name and name.endswith("attn.processor"):
            attn_procs[name] = RegionalStableDiffusion3AttnProcessor2_0()
        else:
            attn_procs[name] = pipeline.transformer.attn_processors[name]
    pipeline.transformer.set_attn_processor(attn_procs)
    
    # example regional prompt and mask pairs
    image_width = 1280
    image_height = 768
    num_samples = 1
    num_inference_steps = 24
    guidance_scale = 3.5
    seed = 124
    base_prompt = "An ancient woman stands solemnly holding a blazing torch, while a fierce battle rages in the background, capturing both strength and tragedy in a historical war scene."
    background_prompt = "a photo"
    regional_prompt_mask_pairs = {
        "0": {
            "description": "A dignified woman in ancient robes stands in the foreground, her face illuminated by the torch she holds high. Her expression is one of determination and sorrow, her clothing and appearance reflecting the historical period. The torch casts dramatic shadows across her features, its flames dancing vibrantly against the darkness.",
            "mask": [128, 128, 640, 768]
        }
    }
    # region control settings
    mask_inject_steps = 10
    double_inject_blocks_interval = 1
    single_inject_blocks_interval = 1
    base_ratio = 0.3

    image_width = (image_width // pipeline.vae_scale_factor) * pipeline.vae_scale_factor
    image_height = (image_height // pipeline.vae_scale_factor) * pipeline.vae_scale_factor

    regional_prompts = []
    regional_masks = []
    background_mask = torch.ones((image_height, image_width))

    for region_idx, region in regional_prompt_mask_pairs.items():
        description = region['description']
        mask = region['mask']
        x1, y1, x2, y2 = mask

        mask = torch.zeros((image_height, image_width))
        mask[y1:y2, x1:x2] = 1.0

        background_mask -= mask

        regional_prompts.append(description)
        regional_masks.append(mask)
            
    # if regional masks don't cover the whole image, append background prompt and mask
    if background_mask.sum() > 0:
        regional_prompts.append(background_prompt)
        regional_masks.append(background_mask)

    # setup regional kwargs that pass to the pipeline
    joint_attention_kwargs = {
        'regional_prompts': regional_prompts,
        'regional_masks': regional_masks,
        'double_inject_blocks_interval': double_inject_blocks_interval,
        'single_inject_blocks_interval': single_inject_blocks_interval,
        'base_ratio': base_ratio,
    }
    # generate images
    images = pipeline(
        prompt=base_prompt,
        num_samples=num_samples,
        width=image_width, height=image_height,
        mask_inject_steps=mask_inject_steps,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cuda").manual_seed(seed),
        joint_attention_kwargs=joint_attention_kwargs,
    ).images

    for idx, image in enumerate(images):
        image.save(f"output_{idx}.jpg")
