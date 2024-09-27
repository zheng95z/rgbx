import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import gradio as gr
import torch
import torchvision
from diffusers import DDIMScheduler
from load_image import load_exr_image, load_ldr_image
from pipeline_rgb2x import StableDiffusionAOVMatEstPipeline

current_directory = os.path.dirname(os.path.abspath(__file__))


def get_rgb2x_demo():
    # Load pipeline
    pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
        "zheng95z/rgb-to-x",
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
        photo,
        seed,
        inference_step,
        num_samples,
    ):
        generator = torch.Generator(device="cuda").manual_seed(seed)

        if photo.name.endswith(".exr"):
            photo = load_exr_image(photo.name, tonemaping=True, clamp=True).to("cuda")
        elif (
            photo.name.endswith(".png")
            or photo.name.endswith(".jpg")
            or photo.name.endswith(".jpeg")
        ):
            photo = load_ldr_image(photo.name, from_srgb=True).to("cuda")

        # Check if the width and height are multiples of 8. If not, crop it using torchvision.transforms.CenterCrop
        old_height = photo.shape[1]
        old_width = photo.shape[2]
        new_height = old_height
        new_width = old_width
        radio = old_height / old_width
        max_side = 1000
        if old_height > old_width:
            new_height = max_side
            new_width = int(new_height / radio)
        else:
            new_width = max_side
            new_height = int(new_width * radio)

        if new_width % 8 != 0 or new_height % 8 != 0:
            new_width = new_width // 8 * 8
            new_height = new_height // 8 * 8

        photo = torchvision.transforms.Resize((new_height, new_width))(photo)

        required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
        prompts = {
            "albedo": "Albedo (diffuse basecolor)",
            "normal": "Camera-space Normal",
            "roughness": "Roughness",
            "metallic": "Metallicness",
            "irradiance": "Irradiance (diffuse lighting)",
        }

        return_list = []
        for i in range(num_samples):
            for aov_name in required_aovs:
                prompt = prompts[aov_name]
                generated_image = pipe(
                    prompt=prompt,
                    photo=photo,
                    num_inference_steps=inference_step,
                    height=new_height,
                    width=new_width,
                    generator=generator,
                    required_aovs=[aov_name],
                ).images[0][0]

                generated_image = torchvision.transforms.Resize(
                    (old_height, old_width)
                )(generated_image)

                generated_image = (generated_image, f"Generated {aov_name} {i}")
                return_list.append(generated_image)

        return return_list

    block = gr.Blocks()
    with block:
        with gr.Row():
            gr.Markdown("## Model RGB -> X (Realistic image -> Intrinsic channels)")
        with gr.Row():
            # Input side
            with gr.Column():
                gr.Markdown("### Given Image")
                photo = gr.File(label="Photo", file_types=[".exr", ".png", ".jpg"])

                gr.Markdown("### Parameters")
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
                        maximum=100,
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
            photo,
            seed,
            inference_step,
            num_samples,
        ]
        run_button.click(fn=callback, inputs=inputs, outputs=result_gallery, queue=True)

    return block


if __name__ == "__main__":
    demo = get_rgb2x_demo()
    demo.queue(max_size=1)
    demo.launch()
