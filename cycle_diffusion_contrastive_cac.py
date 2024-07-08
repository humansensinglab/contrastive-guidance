import requests
import torch
from PIL import Image
from io import BytesIO
from typing import Optional, Union, Tuple, List, Callable, Dict

from diffusers import CycleDiffusionPipeline, DDIMScheduler

import ptp_utils
import seq_aligner
import torch.nn.functional as nnf
import numpy as np
import abc

MAX_NUM_WORDS = 77
NUM_DIFFUSION_STEPS = 50

# load the pipeline
device = "cuda"
# model_id_or_path, img_size = "CompVis/stable-diffusion-v1-4", 512
# model_id_or_path, img_size = "runwayml/stable-diffusion-v1-5", 512
model_id_or_path, img_size = "stabilityai/stable-diffusion-2-base", 512
# model_id_or_path, img_size = "stabilityai/stable-diffusion-2", 768
scheduler = DDIMScheduler.from_config(model_id_or_path, subfolder="scheduler")

if True:
    torch_dtype = torch.float16
    pipe = CycleDiffusionPipeline.from_pretrained(
        model_id_or_path,
        revision="fp16",
        torch_dtype=torch.float16,
        scheduler=scheduler,
    )
if False:
    torch_dtype = torch.float32
    pipe = CycleDiffusionPipeline.from_pretrained(
        model_id_or_path,
        scheduler=scheduler,
    )

print(pipe.scheduler)
tokenizer = pipe.tokenizer
# or download via git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
# and pass `model_id_or_path="./stable-diffusion-v1-4"`.
pipe = pipe.to(device)


class LocalBlend:

    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).to(torch_dtype)
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        attn = self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):

    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device).to(torch_dtype)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device).to(torch_dtype)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                                                                                     Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch_dtype)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer


def main():

    if True:
        # let's download an initial image
        url = "https://raw.githubusercontent.com/ChenWu98/cycle-diffusion/main/data/dalle2/An%20astronaut%20riding%20a%20horse.png"
        response = requests.get(url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((img_size, img_size))
        init_image.save("horse.png")

        # Note: when using CAC controller, the two prompts should
        # (1) differ in only one token, or
        # (2) one prompt is a subsequence of the other.
        source_prompt = "An astronaut riding a horse"
        prompt = "An astronaut riding a elephant"

        # create the CAC controller.
        controller = AttentionReplace([source_prompt, prompt],
                                      NUM_DIFFUSION_STEPS,
                                      cross_replace_steps=1.0,
                                      self_replace_steps=0.4,
                                      )
        source_controller = AttentionRefine([source_prompt, source_prompt],
                                            NUM_DIFFUSION_STEPS,
                                            cross_replace_steps=1.0,
                                            self_replace_steps=0.4,
                                            )
        attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, controller)
        source_attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, source_controller)
        empty_attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, EmptyControl())

        # call the pipeline
        torch.manual_seed(0)
        images = pipe(prompt=prompt,
                      source_prompt=source_prompt,
                      init_image=init_image,
                      num_inference_steps=NUM_DIFFUSION_STEPS,
                      eta=0.1,
                      strength=0.9,
                      guidance_scale=1,
                      source_guidance_scale=1,
                      contrastive_guidance_scale=1,  # TODO
                      attention_control=attention_control,
                      source_attention_control=source_attention_control,
                      empty_attention_control=empty_attention_control,
                      ).images

        images[0].save("horse_to_elephant_contrastive_cac.png")

    if True:
        # let's download an initial image
        url = "https://raw.githubusercontent.com/ChenWu98/cycle-diffusion/main/data/dalle2/A%20black%20colored%20car.png"
        response = requests.get(url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((img_size, img_size))
        init_image.save("black.png")

        # Note: when using CAC controller, the two prompts should
        # (1) differ in only one token, or
        # (2) one prompt is a subsequence of the other.
        source_prompt = "A black colored car"
        prompt = "A blue colored car"

        # create the CAC controller.
        controller = AttentionReplace([source_prompt, prompt],
                                      NUM_DIFFUSION_STEPS,
                                      cross_replace_steps=0.8,
                                      self_replace_steps=0.4,
                                      )
        source_controller = AttentionRefine([source_prompt, source_prompt],
                                            NUM_DIFFUSION_STEPS,
                                            cross_replace_steps=1.0,
                                            self_replace_steps=0.4,
                                            )
        attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, controller)
        source_attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, source_controller)
        empty_attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, EmptyControl())

        # call the pipeline
        torch.manual_seed(0)
        images = pipe(prompt=prompt,
                      source_prompt=source_prompt,
                      init_image=init_image,
                      num_inference_steps=NUM_DIFFUSION_STEPS,
                      eta=0.1,
                      strength=0.95,
                      guidance_scale=1,
                      source_guidance_scale=1,
                      contrastive_guidance_scale=4,  # TODO
                      attention_control=attention_control,
                      source_attention_control=source_attention_control,
                      empty_attention_control=empty_attention_control,
                      ).images

        images[0].save("black_to_blue_contrastive_cac.png")

    if True:
        # let's download an initial image
        url = "https://raw.githubusercontent.com/ChenWu98/cycle-diffusion/main/data/dalle2/A%20black%20colored%20car.png"
        response = requests.get(url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((img_size, img_size))
        init_image.save("black.png")

        # Note: when using CAC controller, the two prompts should
        # (1) differ in only one token, or
        # (2) one prompt is a subsequence of the other.
        source_prompt = "A black colored car"
        prompt = "A red colored car"

        # create the CAC controller.
        controller = AttentionReplace([source_prompt, prompt],
                                      NUM_DIFFUSION_STEPS,
                                      cross_replace_steps=0.8,
                                      self_replace_steps=0.4,
                                      )
        source_controller = AttentionRefine([source_prompt, source_prompt],
                                            NUM_DIFFUSION_STEPS,
                                            cross_replace_steps=1.0,
                                            self_replace_steps=0.4,
                                            )
        attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, controller)
        source_attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, source_controller)
        empty_attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, EmptyControl())

        # call the pipeline
        torch.manual_seed(0)
        images = pipe(prompt=prompt,
                      source_prompt=source_prompt,
                      init_image=init_image,
                      num_inference_steps=NUM_DIFFUSION_STEPS,
                      eta=0.1,
                      strength=0.95,
                      guidance_scale=1,
                      source_guidance_scale=1,
                      contrastive_guidance_scale=5,  # TODO
                      attention_control=attention_control,
                      source_attention_control=source_attention_control,
                      empty_attention_control=empty_attention_control,
                      ).images

        images[0].save("black_to_red_contrastive_cac.png")

    if True:
        # let's download an initial image
        url = "https://raw.githubusercontent.com/ChenWu98/cycle-diffusion/main/data/my_collection/Sun_Yat-sen_Mausoleum_3.png"
        response = requests.get(url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((img_size, img_size))
        init_image.save("aerial_autumn.png")

        # Note: when using CAC controller, the two prompts should
        # (1) differ in only one token, or
        # (2) one prompt is a subsequence of the other.
        source_prompt = "An aerial view of autumn scene."
        prompt = "An aerial view of winter scene."

        # create the CAC controller.
        controller = AttentionReplace([source_prompt, prompt],
                                      NUM_DIFFUSION_STEPS,
                                      cross_replace_steps=0.8,
                                      self_replace_steps=0.4,
                                      )
        source_controller = AttentionRefine([source_prompt, source_prompt],
                                            NUM_DIFFUSION_STEPS,
                                            cross_replace_steps=1.0,
                                            self_replace_steps=0.4,
                                            )
        attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, controller)
        source_attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, source_controller)
        empty_attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, EmptyControl())

        # call the pipeline
        torch.manual_seed(0)
        images = pipe(prompt=prompt,
                      source_prompt=source_prompt,
                      init_image=init_image,
                      num_inference_steps=NUM_DIFFUSION_STEPS,
                      eta=0.1,
                      strength=0.9,
                      guidance_scale=1,
                      source_guidance_scale=1,
                      contrastive_guidance_scale=3,  # TODO
                      attention_control=attention_control,
                      source_attention_control=source_attention_control,
                      empty_attention_control=empty_attention_control,
                      ).images

        images[0].save("aerial_autumn_to_winter_contrastive_cac.png")

    if True:
        # let's download an initial image
        url = "https://raw.githubusercontent.com/ChenWu98/cycle-diffusion/main/data/dalle2/A%20green%20apple%20and%20a%20black%20backpack.png"
        response = requests.get(url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((img_size, img_size))
        init_image.save("apple_backpack.png")

        # Note: when using CAC controller, the two prompts should
        # (1) differ in only one token, or
        # (2) one prompt is a subsequence of the other.
        source_prompt = "A green apple and a black backpack on the floor."
        prompt = "A red apple and a black backpack on the floor."

        # create the CAC controller.
        controller = AttentionReplace([source_prompt, prompt],
                                      NUM_DIFFUSION_STEPS,
                                      cross_replace_steps=0.8,
                                      self_replace_steps=0.4,
                                      )
        source_controller = AttentionRefine([source_prompt, source_prompt],
                                            NUM_DIFFUSION_STEPS,
                                            cross_replace_steps=1.0,
                                            self_replace_steps=0.4,
                                            )
        attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, controller)
        source_attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, source_controller)
        empty_attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, EmptyControl())

        # call the pipeline
        torch.manual_seed(0)
        images = pipe(prompt=prompt,
                      source_prompt=source_prompt,
                      init_image=init_image,
                      num_inference_steps=NUM_DIFFUSION_STEPS,
                      eta=0.1,
                      strength=0.9,
                      guidance_scale=1,
                      source_guidance_scale=1,
                      contrastive_guidance_scale=4,  # TODO
                      attention_control=attention_control,
                      source_attention_control=source_attention_control,
                      empty_attention_control=empty_attention_control,
                      ).images

        images[0].save("apple_backpack_red_contrastive_cac.png")

    if True:
        # let's try another example
        url = "https://raw.githubusercontent.com/ChenWu98/cycle-diffusion/main/data/papers/flower_hotel.png"
        response = requests.get(url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((img_size, img_size))
        init_image.save("red_flower_hotel.png")

        # Note: when using CAC controller, the two prompts should
        # (1) differ in only one token, or
        # (2) one prompt is a subsequence of the other.
        source_prompt = "A hotel room with red flowers on the bed."
        prompt = "A hotel room with blue flowers on the bed."

        # create the CAC controller.
        controller = AttentionReplace([source_prompt, prompt],
                                      NUM_DIFFUSION_STEPS,
                                      cross_replace_steps=0.8,
                                      self_replace_steps=0.4,
                                      )
        source_controller = AttentionRefine([source_prompt, source_prompt],
                                            NUM_DIFFUSION_STEPS,
                                            cross_replace_steps=1.0,
                                            self_replace_steps=0.4,
                                            )
        attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, controller)
        source_attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, source_controller)
        empty_attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, EmptyControl())

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
                     attention_control=attention_control,
                     source_attention_control=source_attention_control,
                     empty_attention_control=empty_attention_control,
                     ).images[0]

        image.save("red_flower_to_yellow_contrastive_cac.png")

    if True:
        # let's download an initial image
        url = "https://raw.githubusercontent.com/ChenWu98/cycle-diffusion/main/data/dalle2/A%20green%20apple%20and%20a%20black%20backpack.png"
        response = requests.get(url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((img_size, img_size))
        init_image.save("apple_backpack.png")

        # Note: when using CAC controller, the two prompts should
        # (1) differ in only one token, or
        # (2) one prompt is a subsequence of the other.
        source_prompt = "A green apple and a black backpack on the floor."
        prompt = "Two green apple and a black backpack on the floor."

        # create the CAC controller.
        controller = AttentionReplace([source_prompt, prompt],
                                      NUM_DIFFUSION_STEPS,
                                      cross_replace_steps=0.1,
                                      self_replace_steps=0.1,
                                      )
        source_controller = AttentionRefine([source_prompt, source_prompt],
                                            NUM_DIFFUSION_STEPS,
                                            cross_replace_steps=1.0,
                                            self_replace_steps=0.4,
                                            )
        attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, controller)
        source_attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, source_controller)
        empty_attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, EmptyControl())

        # call the pipeline
        torch.manual_seed(0)
        images = pipe(prompt=prompt,
                      source_prompt=source_prompt,
                      init_image=init_image,
                      num_inference_steps=NUM_DIFFUSION_STEPS,
                      eta=0.1,
                      strength=0.9,
                      guidance_scale=1,
                      source_guidance_scale=1,
                      contrastive_guidance_scale=3,  # TODO
                      attention_control=attention_control,
                      source_attention_control=source_attention_control,
                      empty_attention_control=empty_attention_control,
                      ).images

        images[0].save("apple_backpack_two_contrastive_cac.png")


if __name__ == "__main__":
    main()
