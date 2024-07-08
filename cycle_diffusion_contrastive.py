import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import CycleDiffusionPipeline, DDIMScheduler


NUM_DIFFUSION_STEPS = 50


def main():
    # load the scheduler. CycleDiffusion only supports stochastic samplers.
    # scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
    #                           num_train_timesteps=1000, clip_sample=False, set_alpha_to_one=False)

    # load the pipeline
    # make sure you're logged in with `huggingface-cli login`
    # model_id_or_path, img_size = "CompVis/stable-diffusion-v1-4", 512
    # model_id_or_path, img_size = "runwayml/stable-diffusion-v1-5", 512
    model_id_or_path, img_size = "stabilityai/stable-diffusion-2-base", 512
    # model_id_or_path, img_size = "stabilityai/stable-diffusion-2", 768
    scheduler = DDIMScheduler.from_config(model_id_or_path, subfolder="scheduler")
    pipe = CycleDiffusionPipeline.from_pretrained(model_id_or_path, revision="fp16",
                                                  torch_dtype=torch.float16, scheduler=scheduler).to("cuda")

    if True:
        # let's download an initial image
        url = "https://raw.githubusercontent.com/ChenWu98/cycle-diffusion/main/data/dalle2/An%20astronaut%20riding%20a%20horse.png"
        response = requests.get(url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((img_size, img_size))
        init_image.save("horse.png")

        # let's specify a prompt
        source_prompt = "An astronaut riding a horse"
        prompt = "An astronaut riding an elephant"

        # call the pipeline
        torch.manual_seed(0)
        image = pipe(prompt=prompt,
                     source_prompt=source_prompt,
                     init_image=init_image,
                     num_inference_steps=NUM_DIFFUSION_STEPS,
                     eta=0.1,
                     strength=0.75,
                     guidance_scale=1,
                     source_guidance_scale=1,
                     contrastive_guidance_scale=1,  # TODO
                     ).images[0]

        image.save("horse_to_elephant_contrastive.png")

    if True:
        # let's try another example
        url = "https://raw.githubusercontent.com/ChenWu98/cycle-diffusion/main/data/dalle2/A%20black%20colored%20car.png"
        response = requests.get(url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((img_size, img_size))
        init_image.save("black.png")

        source_prompt = "A black colored car."
        prompt = "A blue colored car."

        # call the pipeline
        torch.manual_seed(0)
        image = pipe(prompt=prompt,
                     source_prompt=source_prompt,
                     init_image=init_image,
                     num_inference_steps=NUM_DIFFUSION_STEPS,
                     eta=0.1,
                     strength=0.85,
                     guidance_scale=1,
                     source_guidance_scale=1,
                     contrastive_guidance_scale=1,  # TODO
                     ).images[0]

        image.save("black_to_blue_contrastive.png")

    if True:
        # let's try another example
        url = "https://raw.githubusercontent.com/ChenWu98/cycle-diffusion/main/data/my_collection/Sun_Yat-sen_Mausoleum_3.png"
        response = requests.get(url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((img_size, img_size))
        init_image.save("aerial_autumn.png")

        source_prompt = "An aerial view of autumn scene."
        prompt = "An aerial view of winter scene."

        # call the pipeline
        torch.manual_seed(0)
        image = pipe(prompt=prompt,
                     source_prompt=source_prompt,
                     init_image=init_image,
                     num_inference_steps=NUM_DIFFUSION_STEPS,
                     eta=0.1,
                     strength=0.9,
                     guidance_scale=1,
                     source_guidance_scale=1,
                     contrastive_guidance_scale=4,  # TODO
                     ).images[0]

        image.save("aerial_autumn_to_winter_contrastive.png")

    if True:
        # let's try another example
        url = "https://raw.githubusercontent.com/ChenWu98/cycle-diffusion/main/data/dalle2/A%20green%20apple%20and%20a%20black%20backpack.png"
        response = requests.get(url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((img_size, img_size))
        init_image.save("apple_backpack.png")

        source_prompt = "A green apple and a black backpack on the floor."
        prompt = "A red apple and a black backpack on the floor."

        # call the pipeline
        torch.manual_seed(0)
        image = pipe(prompt=prompt,
                     source_prompt=source_prompt,
                     init_image=init_image,
                     num_inference_steps=NUM_DIFFUSION_STEPS,
                     eta=0.1,
                     strength=0.9,
                     guidance_scale=1,
                     source_guidance_scale=1,
                     contrastive_guidance_scale=5,  # TODO
                     ).images[0]

        image.save("apple_backpack_red_contrastive.png")

    if True:
        # let's try another example
        url = "https://raw.githubusercontent.com/ChenWu98/cycle-diffusion/main/data/papers/flower_hotel.png"
        response = requests.get(url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((img_size, img_size))
        init_image.save("red_flower_hotel.png")

        source_prompt = "A hotel room with red flowers on the bed."
        prompt = "A hotel room with a cat sitting on the bed."

        # call the pipeline
        torch.manual_seed(0)
        image = pipe(prompt=prompt,
                     source_prompt=source_prompt,
                     init_image=init_image,
                     num_inference_steps=NUM_DIFFUSION_STEPS,
                     eta=0.1,
                     strength=0.8,
                     guidance_scale=1,
                     source_guidance_scale=1,
                     contrastive_guidance_scale=3,  # TODO
                     ).images[0]

        image.save("red_flower_to_cat_contrastive.png")

    if True:
        # let's try another example
        url = "https://raw.githubusercontent.com/ChenWu98/cycle-diffusion/main/data/papers/flower_hotel.png"
        response = requests.get(url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((img_size, img_size))
        init_image.save("red_flower_hotel.png")

        source_prompt = "A hotel room with red flowers on the bed."
        prompt = "A hotel room with blue flowers on the bed."

        # call the pipeline
        torch.manual_seed(0)
        image = pipe(prompt=prompt,
                     source_prompt=source_prompt,
                     init_image=init_image,
                     num_inference_steps=NUM_DIFFUSION_STEPS,
                     eta=0.1,
                     strength=0.9,
                     guidance_scale=1,
                     source_guidance_scale=1,
                     contrastive_guidance_scale=4,  # TODO
                     ).images[0]

        image.save("red_flower_to_yellow_contrastive.png")

    if True:
        # let's try another example
        url = "https://raw.githubusercontent.com/ChenWu98/cycle-diffusion/main/data/dalle2/A%20green%20apple%20and%20a%20black%20backpack.png"
        response = requests.get(url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((img_size, img_size))
        init_image.save("apple_backpack.png")

        source_prompt = "A green apple and a black backpack on the floor."
        prompt = "Two green apples and a black backpack on the floor."

        # call the pipeline
        torch.manual_seed(0)
        image = pipe(prompt=prompt,
                     source_prompt=source_prompt,
                     init_image=init_image,
                     num_inference_steps=NUM_DIFFUSION_STEPS,
                     eta=0.1,
                     strength=0.9,
                     guidance_scale=1,
                     source_guidance_scale=1,
                     contrastive_guidance_scale=3,  # TODO
                     ).images[0]

        image.save("apple_backpack_two_contrastive.png")


if __name__ == "__main__":
    main()
