import argparse
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import gradio as gr
import numpy as np
import torch
from diffusers import DDIMScheduler
from load_image import load_exr_image, load_ldr_image
from pipeline_x2rgb_inpainting import StableDiffusionAOVDropoutPipeline

current_directory = os.path.dirname(os.path.abspath(__file__))


def get_x2rgb_demo():
    # Load pipeline
    pipe = StableDiffusionAOVDropoutPipeline.from_pretrained(
        "zheng95z/x-to-rgb-inpainting",
        torch_dtype=torch.float16,
        cache_dir=os.path.join(current_directory, "model_cache"),
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")

    # Augmentation
    def callback(
        albedo,
        normal,
        roughness,
        metallic,
        irradiance,
        mask,
        photo,
        prompt,
        seed,
        inference_step,
        num_samples,
        guidance_scale,
        image_guidance_scale,
    ):
        if albedo is None:
            albedo_image = None
        elif albedo.name.endswith(".exr"):
            albedo_image = load_exr_image(albedo.name, clamp=True).to("cuda")
        elif (
            albedo.name.endswith(".png")
            or albedo.name.endswith(".jpg")
            or albedo.name.endswith(".jpeg")
        ):
            albedo_image = load_ldr_image(albedo.name, from_srgb=True).to("cuda")

        if normal is None:
            normal_image = None
        elif normal.name.endswith(".exr"):
            normal_image = load_exr_image(normal.name, normalize=True).to("cuda")
        elif (
            normal.name.endswith(".png")
            or normal.name.endswith(".jpg")
            or normal.name.endswith(".jpeg")
        ):
            normal_image = load_ldr_image(normal.name, normalize=True).to("cuda")

        if roughness is None:
            roughness_image = None
        elif roughness.name.endswith(".exr"):
            roughness_image = load_exr_image(roughness.name, clamp=True).to("cuda")
        elif (
            roughness.name.endswith(".png")
            or roughness.name.endswith(".jpg")
            or roughness.name.endswith(".jpeg")
        ):
            roughness_image = load_ldr_image(roughness.name, clamp=True).to("cuda")

        if metallic is None:
            metallic_image = None
        elif metallic.name.endswith(".exr"):
            metallic_image = load_exr_image(metallic.name, clamp=True).to("cuda")
        elif (
            metallic.name.endswith(".png")
            or metallic.name.endswith(".jpg")
            or metallic.name.endswith(".jpeg")
        ):
            metallic_image = load_ldr_image(metallic.name, clamp=True).to("cuda")

        if irradiance is None:
            irradiance_image = None
        elif irradiance.name.endswith(".exr"):
            irradiance_image = load_exr_image(
                irradiance.name, tonemaping=True, clamp=True
            ).to("cuda")
        elif (
            irradiance.name.endswith(".png")
            or irradiance.name.endswith(".jpg")
            or irradiance.name.endswith(".jpeg")
        ):
            irradiance_image = load_ldr_image(
                irradiance.name, from_srgb=True, clamp=True
            ).to("cuda")

        generator = torch.Generator(device="cuda").manual_seed(seed)

        height = 768
        width = 768
        # Check if any of the given images are not None
        images = [
            albedo_image,
            normal_image,
            roughness_image,
            metallic_image,
            irradiance_image,
        ]

        assert photo is not None
        assert mask is not None
        if mask.name.endswith(".exr"):
            mask = load_exr_image(mask.name, clamp=True).to("cuda")[0:1]
        elif (
            mask.name.endswith(".png")
            or mask.name.endswith(".jpg")
            or mask.name.endswith(".jpeg")
        ):
            mask = load_ldr_image(mask.name).to("cuda")[0:1]

        mask = 1.0 - mask

        if photo.name.endswith(".exr"):
            photo = load_exr_image(photo.name, tonemaping=True, clamp=True).to("cuda")
        elif (
            photo.name.endswith(".png")
            or photo.name.endswith(".jpg")
            or photo.name.endswith(".jpeg")
        ):
            photo = load_ldr_image(photo.name, from_srgb=True).to("cuda")

        for img in images:
            if img is not None:
                height = img.shape[1]
                width = img.shape[2]
                break

        masked_photo = photo * mask

        required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
        return_list = []
        for i in range(num_samples):
            res = pipe(
                prompt=prompt,
                albedo=albedo_image,
                normal=normal_image,
                roughness=roughness_image,
                metallic=metallic_image,
                irradiance=irradiance_image,
                mask=mask,
                masked_image=masked_photo,
                photo=photo,
                num_inference_steps=inference_step,
                height=height,
                width=width,
                generator=generator,
                required_aovs=required_aovs,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                guidance_rescale=0.7,
                output_type="np",
            ).images
            generated_image = res[0][0]
            masked_photo_vae = res[1][0]
            photo_vae = res[2][0]
            generated_image = (generated_image, f"Generated Image {i}")
            return_list.append(generated_image)

        masked_photo_vae = (masked_photo_vae, "Masked photo")
        photo_vae = (photo_vae, "Photo")
        return_list.append(masked_photo_vae)
        return_list.append(photo_vae)

        return return_list

    block = gr.Blocks()
    with block:
        with gr.Row():
            gr.Markdown(
                "## Model X -> RGB (Intrinsic channels -> realistic image) inpainting"
            )
        with gr.Row():
            # Input side
            with gr.Column():
                gr.Markdown("### Given intrinsic channels")
                albedo = gr.File(label="Albedo", file_types=[".exr", ".png", ".jpg"])
                normal = gr.File(label="Normal", file_types=[".exr", ".png", ".jpg"])
                roughness = gr.File(
                    label="Roughness", file_types=[".exr", ".png", ".jpg"]
                )
                metallic = gr.File(
                    label="Metallic", file_types=[".exr", ".png", ".jpg"]
                )
                irradiance = gr.File(
                    label="Irradiance", file_types=[".exr", ".png", ".jpg"]
                )
                mask = gr.File(label="Mask", file_types=[".exr", ".png", ".jpg"])
                photo = gr.File(label="Photo", file_types=[".exr", ".png", ".jpg"])

                gr.Markdown("### Parameters")
                prompt = gr.Textbox(label="Prompt")
                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=False):
                    seed = gr.Slider(
                        label="Seed",
                        minimum=-1,
                        maximum=2147483647,
                        step=1,
                        randomize=True,
                    )
                    inference_step = gr.Slider(
                        label="Inference Step",
                        minimum=1,
                        maximum=200,
                        step=1,
                        value=50,
                    )
                    num_samples = gr.Slider(
                        label="Samples",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=1,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=7.5,
                    )
                    image_guidance_scale = gr.Slider(
                        label="Image Guidance Scale",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=1.5,
                    )

            # Output side
            with gr.Column():
                gr.Markdown("### Output Gallery")
                result_gallery = gr.Gallery(
                    label="Output",
                    show_label=False,
                    elem_id="gallery",
                    columns=2,
                )

        inputs = [
            albedo,
            normal,
            roughness,
            metallic,
            irradiance,
            mask,
            photo,
            prompt,
            seed,
            inference_step,
            num_samples,
            guidance_scale,
            image_guidance_scale,
        ]
        run_button.click(fn=callback, inputs=inputs, outputs=result_gallery, queue=True)

    return block


if __name__ == "__main__":
    demo = get_x2rgb_demo()
    demo.queue(max_size=1)
    demo.launch()
