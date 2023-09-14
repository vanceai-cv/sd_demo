import os
import json

from PIL import Image
from sdwebuiapi import webuiapi

class ModuleImg2sr:
    def __init__(self):
        self.model_name = 'v2-1_512-ema-pruned.ckpt'
        self.sr_model = 'stablesr_webui_sd-v2-1-512-ema-000117.ckpt'

    def single_img2img(self, image, width, height):
        
        api = webuiapi.WebUIApi(host='158.255.7.19', port=7860)

        options = {}
        options['sd_model_checkpoint'] = self.model_name
        options['sd_vae'] = 'vqgan_cfw_00011_vae_only.ckpt'
        api.set_options(options)

        sampler_name = 'Euler a'
        steps = 25
        denoising_strength = 0.75
        Pure_Noise = False
        seed = -1
        script_args = []
        script_args =[self.sr_model,1.0,Pure_Noise,'Wavelet',False]
       
        img2img_args = {
            # custom params
            "images": [image],  # list of PIL Image
            "sampler_name": sampler_name,
            "steps": steps,
            "width": width,
            "height": height,
            "cfg_scale": 2,
            "denoising_strength": denoising_strength,
            "seed": seed,

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
            "restore_faces": True,
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
            "script_args": script_args,  # List of arguments for the script "script_name"
            "sampler_index": None,  # deprecated: use sampler_name
            "include_init_images": False,
            "script_name": 'StableSR',
            "send_images": True,
            "save_images": False,
            "alwayson_scripts": {},
            "use_deprecated_controlnet": False
        }
        response = api.img2img(**img2img_args)

        if len(response.images) == 2:
            output = response.images[1]
        else:
            output = response.image
        return output

    def inference(self, config):
        try:
            src_path = config['src_path']
            dst_path = config['dst_path']
            dst_name = config['dst_name']
            out_path = os.path.join(dst_path, dst_name)
            self.model_name = config['model_name']
            self.sr_model = config['sr_model']

            image = Image.open(src_path)             
            w, h = image.size
            width = w
            height = h

            if (w > h and w/h > 2) or (h > w and h/w > 2):
                if (h >= w and h < 1024) or (w >= h and w < 1024):
                    height = 1024
                    scale_percent = h/1024
                    width = round(w/scale_percent)        
                if h >= w and h >= 1024:
                    height = 1024
                    scale_percent = h/1024
                    width = round(w/scale_percent)
                if w >= h and w >= 1024:
                    width = 1024
                    scale_percent = w/1024
                    height = round(h/scale_percent)
            else:
                if (h >= w and h < 1024) or (w >= h and w < 1024):
                    height = 1024
                    scale_percent = h/1024
                    width = round(w/scale_percent)        
                if h >= w and h >= 1024:
                    width = 1024
                    scale_percent = w/1024
                    height = round(h/scale_percent)
                if w >= h and w >= 1024:
                    height = 1024
                    scale_percent = h/1024
                    width = round(w/scale_percent)

            image = image.resize((width, height))
            dst_img = self.single_img2img(image, width, height)
            dst_img.save(out_path,compress_level=1)   

        except Exception as e:
            print("Exception in inference", exc_info=True)

if __name__ == '__main__':
    cfg_path = 'jparam/img2sr.json'
    with open(cfg_path, 'rb') as fp:
        config = json.load(fp)

    model = ModuleImg2sr()
    bret = model.inference(config)