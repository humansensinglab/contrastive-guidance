import torch
from PIL import Image

from diffusers import StableDiffusionContrastivePipeline, DPMSolverMultistepScheduler, PNDMScheduler


def main():
    # load the pipeline
    # make sure you're logged in with `huggingface-cli login`
    # model_id_or_path = "CompVis/stable-diffusion-v1-4"
    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    # model_id_or_path = "stabilityai/stable-diffusion-2-base"
    # model_id_or_path = "stabilityai/stable-diffusion-2"
    if True:  # TODO
        scheduler = DPMSolverMultistepScheduler.from_config(model_id_or_path, subfolder="scheduler")
        num_inference_steps = 20
    if False:  # TODO
        scheduler = PNDMScheduler.from_config(model_id_or_path, subfolder="scheduler")
        num_inference_steps = 50

    pipe = StableDiffusionContrastivePipeline.from_pretrained(model_id_or_path, revision="fp16", scheduler=scheduler,
                                                              torch_dtype=torch.float16).to("cuda")
    print(pipe.scheduler)

    # Continuous, rig-like control
    if False:
        # let's specify a prompt
        prompt = "A photo of a fox."

        # let's specify contrastive prompts
        positive_prompt1 = "A photo of a cute fox."
        baseline_prompt1 = "A photo of a fox."
        contrastive_prompts = [
            [positive_prompt1, baseline_prompt1],
        ]

        # Enumerate contrastive prompts
        images = []
        for contrastive_scale in [-20, -15, -10, -5, 0, 5, 10, 15, 20]:
            contrastive_scales = [contrastive_scale]

            # call the pipeline
            generator = torch.Generator(device="cuda").manual_seed(0)
            image = pipe(prompt=prompt,
                         contrastive_prompts=contrastive_prompts,
                         contrastive_scales=contrastive_scales,
                         num_inference_steps=num_inference_steps,
                         guidance_scale=7.5,
                         generator=generator,
                         ).images[0]
            images.append(image)

        # save the images as a grid
        h, w = images[0].size
        grid = Image.new("RGB", (w * len(images), h))
        for i, image in enumerate(images):
            grid.paste(image, (w * i, 0))
        grid.save("fox.png")

    # Continuous, rig-like control
    if False:
        # let's specify a prompt
        prompt = "A photo of a waterfall."

        # let's specify contrastive prompts
        positive_prompt1 = "A beautiful photo of a waterfall."
        baseline_prompt1 = "A photo of a waterfall."
        contrastive_prompts = [
            [positive_prompt1, baseline_prompt1],
        ]

        # Enumerate contrastive prompts
        images = []
        for contrastive_scale in [-20, -15, -10, -5, 0, 5, 10, 15, 20]:
            contrastive_scales = [contrastive_scale]

            # call the pipeline
            generator = torch.Generator(device="cuda").manual_seed(0)
            image = pipe(prompt=prompt,
                         contrastive_prompts=contrastive_prompts,
                         contrastive_scales=contrastive_scales,
                         num_inference_steps=num_inference_steps,
                         guidance_scale=7.5,
                         generator=generator,
                         ).images[0]
            images.append(image)

        # save the images as a grid
        h, w = images[0].size
        grid = Image.new("RGB", (w * len(images), h))
        for i, image in enumerate(images):
            grid.paste(image, (w * i, 0))
        grid.save("waterfall.png")

    # Continuous, rig-like control
    if False:
        # let's specify a prompt
        prompt = "A photo of a valley."

        # let's specify contrastive prompts
        positive_prompt1 = "A beautiful photo of a valley."
        baseline_prompt1 = "A photo of a valley."
        contrastive_prompts = [
            [positive_prompt1, baseline_prompt1],
        ]

        # Enumerate contrastive prompts
        images = []
        for contrastive_scale in [-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]:
            contrastive_scales = [contrastive_scale]

            # call the pipeline
            generator = torch.Generator(device="cuda").manual_seed(0)
            image = pipe(prompt=prompt,
                         contrastive_prompts=contrastive_prompts,
                         contrastive_scales=contrastive_scales,
                         num_inference_steps=num_inference_steps,
                         guidance_scale=7.5,
                         generator=generator,
                         ).images[0]
            images.append(image)

        # save the images as a grid
        h, w = images[0].size
        grid = Image.new("RGB", (w * len(images), h))
        for i, image in enumerate(images):
            grid.paste(image, (w * i, 0))
        grid.save("valley.png")

    # Continuous, rig-like control
    if True:
        # let's specify a prompt
        prompt = "A crowded park in an autumn morning."

        # let's specify contrastive prompts
        positive_prompt1 = "A crowded park in an autumn morning."
        baseline_prompt1 = "A empty park in an autumn morning."
        contrastive_prompts = [
            [positive_prompt1, baseline_prompt1],
        ]

        # Enumerate contrastive prompts
        images = []
        for contrastive_scale in [-10, -5, 0, 5, 10]:
            contrastive_scales = [contrastive_scale]

            # call the pipeline
            generator = torch.Generator(device="cuda").manual_seed(0)
            image = pipe(prompt=prompt,
                         contrastive_prompts=contrastive_prompts,
                         contrastive_scales=contrastive_scales,
                         num_inference_steps=num_inference_steps,
                         guidance_scale=7.5,
                         strength=0.7,  # TODO
                         generator=generator,
                         ).images[0]
            images.append(image)

        # save the images as a grid
        h, w = images[0].size
        grid = Image.new("RGB", (w * len(images), h))
        for i, image in enumerate(images):
            grid.paste(image, (w * i, 0))
        grid.save("park.png")

    # Continuous, rig-like control
    if False:
        # let's specify a prompt
        prompt = "An oil painting of a couple in formal evening wear going home."

        # let's specify contrastive prompts
        positive_prompt1 = "An oil painting of a couple in formal evening wear going home in a heavy downpour."
        baseline_prompt1 = "An oil painting of a couple in formal evening wear going home in a sunny day."
        contrastive_prompts = [
            [positive_prompt1, baseline_prompt1],
        ]

        # Enumerate contrastive prompts
        images = []
        for contrastive_scale in [-5, -3.75, -2.5, -1.25, 0, 1.25, 2.5, 3.75, 5]:
            contrastive_scales = [contrastive_scale]

            # call the pipeline
            generator = torch.Generator(device="cuda").manual_seed(0)
            image = pipe(prompt=prompt,
                         contrastive_prompts=contrastive_prompts,
                         contrastive_scales=contrastive_scales,
                         num_inference_steps=num_inference_steps,
                         guidance_scale=7.5,
                         generator=generator,
                         ).images[0]
            images.append(image)

        # save the images as a grid
        h, w = images[0].size
        grid = Image.new("RGB", (w * len(images), h))
        for i, image in enumerate(images):
            grid.paste(image, (w * i, 0))
        grid.save("couple.png")

    # Continuous, rig-like control
    if False:
        # let's specify a prompt
        prompt = "A photo of a fox on a street in new york."

        # let's specify contrastive prompts
        positive_prompt1 = "A bright photo of a fox on a street in new york."
        baseline_prompt1 = "A dark photo of a fox on a street in new york."
        contrastive_prompts = [
            [positive_prompt1, baseline_prompt1],
        ]

        # Enumerate contrastive prompts
        images = []
        for contrastive_scale in [-5, -2.5, 0, 2.5, 5]:
            contrastive_scales = [contrastive_scale]

            # call the pipeline
            generator = torch.Generator(device="cuda").manual_seed(0)
            image = pipe(prompt=prompt,
                         contrastive_prompts=contrastive_prompts,
                         contrastive_scales=contrastive_scales,
                         num_inference_steps=num_inference_steps,
                         guidance_scale=7.5,
                         generator=generator,
                         ).images[0]
            images.append(image)

        # save the images as a grid
        h, w = images[0].size
        grid = Image.new("RGB", (w * len(images), h))
        for i, image in enumerate(images):
            grid.paste(image, (w * i, 0))
        grid.save("brightness_fox.png")

    if False:
        # let's specify a prompt
        prompt = "A photo of a fox on a street in new york + ornaments, neon lights, watercolor, pen and ink, intricate line drawings, by yoshitaka amano, ruan jia, kentaro miura, artgerm, post processed, concept art, artstation, matte painting, style by eddie mendoza, raphael lacoste, alex ross."

        # let's specify contrastive prompts
        positive_prompt1 = "A photo of a cute fox on a street in new york + ornaments, neon lights, watercolor, pen and ink, intricate line drawings, by yoshitaka amano, ruan jia, kentaro miura, artgerm, post processed, concept art, artstation, matte painting, style by eddie mendoza, raphael lacoste, alex ross."
        baseline_prompt1 = "A photo of a fox on a street in new york + ornaments, neon lights, watercolor, pen and ink, intricate line drawings, by yoshitaka amano, ruan jia, kentaro miura, artgerm, post processed, concept art, artstation, matte painting, style by eddie mendoza, raphael lacoste, alex ross."
        contrastive_prompts = [
            [positive_prompt1, baseline_prompt1],
        ]

        # Enumerate contrastive prompts
        images = []
        for contrastive_scale in [-5, -2.5, 0, 2.5, 5]:
            contrastive_scales = [contrastive_scale]

            # call the pipeline
            generator = torch.Generator(device="cuda").manual_seed(0)
            image = pipe(prompt=prompt,
                         contrastive_prompts=contrastive_prompts,
                         contrastive_scales=contrastive_scales,
                         num_inference_steps=num_inference_steps,
                         guidance_scale=7.5,
                         strength=0.5,  # TODO
                         generator=generator,
                         ).images[0]
            images.append(image)

        # save the images as a grid
        h, w = images[0].size
        grid = Image.new("RGB", (w * len(images), h))
        for i, image in enumerate(images):
            grid.paste(image, (w * i, 0))
        grid.save("complex_fox.png")

    if False:
        # let's specify a prompt
        prompt = "A green banana on a table."

        # let's specify contrastive prompts
        positive_prompt1 = "A green banana on a table."
        baseline_prompt1 = "A yellow banana on a table."
        contrastive_prompts = [
            [positive_prompt1, baseline_prompt1],
        ]

        # Enumerate contrastive prompts
        images = []
        for contrastive_scale in [-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]:
            contrastive_scales = [contrastive_scale]

            # call the pipeline
            generator = torch.Generator(device="cuda").manual_seed(0)
            image = pipe(prompt=prompt,
                         contrastive_prompts=contrastive_prompts,
                         contrastive_scales=contrastive_scales,
                         num_inference_steps=num_inference_steps,
                         guidance_scale=7.5,
                         strength=0.75,  # TODO
                         generator=generator,
                         ).images[0]
            images.append(image)

        # save the images as a grid
        h, w = images[0].size
        grid = Image.new("RGB", (w * len(images), h))
        for i, image in enumerate(images):
            grid.paste(image, (w * i, 0))
        grid.save("banana.png")

    if False:
        # let's specify a prompt
        prompt = "A short-neck giraffe standing on grass, under the sky. Cartoon."

        # let's specify contrastive prompts
        positive_prompt1 = "A short-neck giraffe standing on grass, under the sky. Cartoon."
        baseline_prompt1 = "A long-neck giraffe standing on grass, under the sky. Cartoon."
        contrastive_prompts = [
            [positive_prompt1, baseline_prompt1],
        ]

        # Enumerate contrastive prompts
        images = []
        for contrastive_scale in [-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]:
            contrastive_scales = [contrastive_scale]

            # call the pipeline
            generator = torch.Generator(device="cuda").manual_seed(0)
            image = pipe(prompt=prompt,
                         contrastive_prompts=contrastive_prompts,
                         contrastive_scales=contrastive_scales,
                         num_inference_steps=num_inference_steps,
                         guidance_scale=7.5,
                         strength=0.6,  # TODO
                         generator=generator,
                         ).images[0]
            images.append(image)

        # save the images as a grid
        h, w = images[0].size
        grid = Image.new("RGB", (w * len(images), h))
        for i, image in enumerate(images):
            grid.paste(image, (w * i, 0))
        grid.save("giraffe.png")


if __name__ == "__main__":
    main()
