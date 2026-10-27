import os
import zipfile
import tempfile

import torch
import gradio as gr
from diffusers import StableDiffusionPipeline

MODEL_ID = "runwayml/stable-diffusion-v1-5"

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe = pipe.to(device)

STYLES = {
    "Realistic": "realistic, ultra detailed, 8k",
    "Anime": "anime style, vibrant colors",
    "Oil Painting": "oil painting, masterpiece",
    "Watercolor": "watercolor painting",
    "Pixel Art": "pixel art",
}


def generate_images(prompt, style, negative_prompt, num_images):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter an image prompt.")

    final_prompt = f"{prompt}, {STYLES[style]}"

    with tempfile.TemporaryDirectory() as temp_dir:
        result = pipe(
            prompt=final_prompt,
            negative_prompt=negative_prompt or None,
            num_images_per_prompt=int(num_images)
        )

        image_paths = []

        for index, image in enumerate(result.images, start=1):
            path = os.path.join(
                temp_dir, f"generated_image_{index}.png"
            )
            image.save(path)
            image_paths.append(path)

        zip_path = os.path.join(temp_dir, "generated_images.zip")

        with zipfile.ZipFile(zip_path, "w") as archive:
            for path in image_paths:
                archive.write(path, os.path.basename(path))

        # Keep files outside the temporary directory for Gradio.
        output_dir = tempfile.mkdtemp(prefix="generated_images_")
        saved_images = []

        for path in image_paths:
            destination = os.path.join(
                output_dir, os.path.basename(path)
            )
            image = result.images[len(saved_images)]
            image.save(destination)
            saved_images.append(destination)

        saved_zip = os.path.join(output_dir, "generated_images.zip")
        with zipfile.ZipFile(saved_zip, "w") as archive:
            for path in saved_images:
                archive.write(path, os.path.basename(path))

        return saved_images, saved_zip


with gr.Blocks(title="AI Image Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Image Generator")
    gr.Markdown("Generate images from your text prompts using Stable Diffusion.")

    prompt = gr.Textbox(
        label="Enter your prompt",
        placeholder="A beautiful mountain landscape at sunrise"
    )

    style = gr.Dropdown(
        choices=list(STYLES.keys()),
        value="Realistic",
        label="Image Style"
    )

    negative_prompt = gr.Textbox(
        label="Negative Prompt",
        placeholder="blurry, low quality, distorted"
    )

    num_images = gr.Slider(
        minimum=1,
        maximum=4,
        value=1,
        step=1,
        label="Number of Images"
    )

    generate_button = gr.Button("Generate Images", variant="primary")

    gallery = gr.Gallery(label="Generated Images", columns=2)
    download_file = gr.File(label="Download Images ZIP")

    generate_button.click(
        fn=generate_images,
        inputs=[prompt, style, negative_prompt, num_images],
        outputs=[gallery, download_file]
    )

if __name__ == "__main__":
    demo.launch()
