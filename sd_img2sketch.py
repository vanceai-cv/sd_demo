import os
import sys
import time
import cv2
import numpy as np
import json

from PIL import Image
from sdwebuiapi import webuiapi

class ModuleImg2sketch:
    def __init__(self):
        self.model_name = 'realisticVisionV20_v20.safetensors'
        self.description = ''

    def single_img2img(self, image, width, height, write_image):
        """
        Important: 

        - When calling the server, call it with these params: --api --ckpt models\Stable-diffusion\neverendingDreamNED_v122BakedVae.safetensors
        - If you are testing ControlNet, don't forget to Check "Enable"

        Params:
        image_path: input image path
        control mode: -1 - no controlnet; 0 - balanced; 1 - prompt is more important; 2 - controlnet is more important

        Part of the params used by the UI:
        Negative prompt: worst quality, low quality:1.4, extra arms, EasyNegative, bad-hands-5, bad-artist
        Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 8, Seed: 2634715279, Size: 512x768, Model hash: ecefb796ff, Model: neverendingDreamNED_v122BakedVae, Denoising strength: 0.75, Version: v1.2.1
        """
        
        api = webuiapi.WebUIApi(host='158.255.7.19', port=7860)

        options = {}
        options['sd_model_checkpoint'] = self.model_name
        #options['sd_vae'] = 'vae-ft-mse-840000-ema-pruned.safetensors'
        api.set_options(options)

        #negative_prompt = "worst quality, low quality:1.4, extra arms, EasyNegative, bad-hands-5, bad-artist"
        negative_prompt = "worst quality, low quality, normal quality, low resolution, ugly, ng_deepnegative_v1_75t, EasyNegative, bad-artist, jpeg artifacts, monochrome, soft filter, blurry image, oversaturated, FastNegativeEmbedding,\
                            nsfw, grayscale watermark, text, title, logo, signature, zombie, sketch\
                            Poorly Rendered face, poorly drawn face, poor facial details,\
                            deformed eyes, crossed eyes,\
                            bad-hands-5, badhandv4, bad hands, poorly drawn hands, poorly rendered hands,\
                            missing fingers, deformed fingers, long fingers, ugly fingers, too many fingers, extra fingers, 6 fingers,  interlocked fingers\
                            multiple limbs, extra limbs, bad anatomy, contorted limbs, deformed body part, long body, disfigured, mutation, missing limbs, skin mole,\
                            realisticvision-negative-embedding,\ (3D, realistic, depth of field, shadow:1.2)"
        sampler_name = 'DPM++ 2M Karras'
        steps = 20

        controlnet_args_canny = {
            "input_image": image,  # Image
            "mask": None,  # Image
            "module": "canny",
            "model": "control_sd15_canny",
            "weight": 1.2,
            "resize_mode": "Crop and Resize",
            "lowvram": False,
            "processor_res": 512,
            "threshold_a": 100,
            "threshold_b": 200,
            "guidance": 1,
            "guidance_start": 0.85,
            "guidance_end": 1.0,
            "control_mode": 2,
            "pixel_perfect": True,
        }

        controlnet_args_normal = {
            "input_image": image,  # Image
            "mask": None,  # Image
            "module": "normal_bae",
            "model": "control_v11p_sd15_normalbae",
            "weight": 1.2,
            "resize_mode": "Crop and Resize",
            "lowvram": False,
            "processor_res": 512,
            "guidance": 1,
            "guidance_start": 0.0,
            "guidance_end": 1.0,
            "control_mode": 2,
            "pixel_perfect": True,
        }
        
        controlnet_unit = []
        cfg_scale = 7
        seed = -1
        restore_faces = True
        denoising_strength = 0.8
        controlnet_unit.append(webuiapi.ControlNetUnit(**controlnet_args_canny))
        controlnet_unit.append(webuiapi.ControlNetUnit(**controlnet_args_normal))

        generated_prompt = "(a line drawing, lineart, linework, black and white style, 2D, outline, outline of hair:1.2)"

        generated_prompt = generated_prompt + self.description

        img2img_args = {
            # custom params
            "images": [write_image],  # list of PIL Image
            "prompt": generated_prompt, #temporary just to make sure this isnt a source of error
            "negative_prompt": negative_prompt,
            "sampler_name": sampler_name,
            "steps": steps,
            "width": width,
            "height": height,
            "cfg_scale": cfg_scale,
            "denoising_strength": denoising_strength,
            "seed": seed,
            "controlnet_units": controlnet_unit,

            # defaults
            "resize_mode": 0,
            "image_cfg_scale": 1.5,
            "mask_image": None,  # PIL Image mask
            "mask_blur": 4,
            "inpainting_fill": 0,
            "inpaint_full_res": True,
            "inpaint_full_res_padding": 0,
            "inpainting_mask_invert": 0,
            "initial_noise_multiplier": 1,
            "styles": [],
            "subseed": -1,
            "subseed_strength": 0,
            "seed_resize_from_h": 0,
            "seed_resize_from_w": 0,
            "batch_size": 1,
            "n_iter": 1,
            "restore_faces": restore_faces,
            "tiling": False,
            "do_not_save_samples": False,
            "do_not_save_grid": False,
            "eta": 1.0,
            "s_churn": 0,
            "s_tmax": 0,
            "s_tmin": 0,
            "s_noise": 1,
            "override_settings": {},
            "override_settings_restore_afterwards": True,
            "script_args": None,  # List of arguments for the script "script_name"
            "sampler_index": None,  # deprecated: use sampler_name
            "include_init_images": False,
            "script_name": None,
            "send_images": True,
            "save_images": False,
            "alwayson_scripts": {},
            "use_deprecated_controlnet": False
        }

        response = api.img2img(**img2img_args)

        output = response.image
        return output

    def inference(self, config):
        try:
            src_path = config['src_path']
            dst_path = config['dst_path']
            dst_name = config['dst_name']
            out_path = os.path.join(dst_path, dst_name)
            self.description = config['description']
            self.model_name = config['model_name']
            image = Image.open(src_path)           
            w, h = image.size
            width = w
            height = h
            scale_percent = 2
            write_image = Image.new('RGB', (w, h), (255, 255, 255))
   
            if h >= w and h >= 512:
                width = 512
                scale_percent = w/512
                height = round(h/scale_percent)
            if w >= h and w >= 512:
                height = 512
                scale_percent = h/512
                width = round(w/scale_percent)
            
            image = image.resize((width, height))
            dst_img = self.single_img2img(image, width, height, write_image)
            dst_img.save(out_path,compress_level=1)

        except Exception as e:
            print("Exception in inference", exc_info=True)

if __name__ == '__main__':
    cfg_path = 'jparam/img2sketch.json'
    with open(cfg_path, 'rb') as fp:
        config = json.load(fp)

    model = ModuleImg2sketch()
    bret = model.inference(config)