import torch
from PIL import Image

from diffusers import StableDiffusionContrastivePipeline, DPMSolverMultistepScheduler, PNDMScheduler

import ptp_utils
import seq_aligner
import torch.nn.functional as nnf
from typing import Optional, Union, Tuple, List, Callable, Dict
import abc

MAX_NUM_WORDS = 77
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

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
tokenizer = pipe.tokenizer


class LocalBlend:

    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).to(x_t.dtype)
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
        self.alpha_layers = alpha_layers.to(device).to(torch_dtype)
        self.threshold = threshold


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
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

    def forward(self, attn, is_cross: bool, place_in_unet: str):
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
            h = attn.shape[0] // self.batch_size
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_replace_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_replace_new
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
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device).to(torch_dtype)
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


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float], Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch_dtype)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer



def main():

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

            # specify the cross-attention control parameters
            contrastive_attention_control_baseline1 = AttentionRefine([prompt, baseline_prompt1],
                                                                      num_inference_steps,
                                                                      cross_replace_steps=1.0,
                                                                      self_replace_steps=0.4,
                                                                      )
            contrastive_attention_control_positive1 = AttentionRefine([prompt, positive_prompt1],
                                                                      num_inference_steps,
                                                                      cross_replace_steps=1.0,
                                                                      self_replace_steps=0.4,
                                                                      )
            contrastive_attention_controls = [
                (
                    lambda pipe: ptp_utils.register_attention_control(pipe, contrastive_attention_control_baseline1),
                    lambda pipe: ptp_utils.register_attention_control(pipe, contrastive_attention_control_positive1)
                ),
            ]
            empty_attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, EmptyControl())

            # call the pipeline
            generator = torch.Generator(device="cuda").manual_seed(0)
            image = pipe(prompt=prompt,
                         contrastive_prompts=contrastive_prompts,
                         contrastive_scales=contrastive_scales,
                         contrastive_attention_controls=contrastive_attention_controls,
                         empty_attention_control=empty_attention_control,
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
        grid.save("fox_cac.png")

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

            # specify the cross-attention control parameters
            contrastive_attention_control_baseline1 = AttentionRefine([prompt, baseline_prompt1],
                                                                      num_inference_steps,
                                                                      cross_replace_steps=1.0,
                                                                      self_replace_steps=0.4,
                                                                      )
            contrastive_attention_control_positive1 = AttentionRefine([prompt, positive_prompt1],
                                                                      num_inference_steps,
                                                                      cross_replace_steps=1.0,
                                                                      self_replace_steps=0.4,
                                                                      )
            contrastive_attention_controls = [
                (
                    lambda pipe: ptp_utils.register_attention_control(pipe, contrastive_attention_control_baseline1),
                    lambda pipe: ptp_utils.register_attention_control(pipe, contrastive_attention_control_positive1)
                ),
            ]
            empty_attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, EmptyControl())

            # call the pipeline
            generator = torch.Generator(device="cuda").manual_seed(0)
            image = pipe(prompt=prompt,
                         contrastive_prompts=contrastive_prompts,
                         contrastive_scales=contrastive_scales,
                         contrastive_attention_controls=contrastive_attention_controls,
                         empty_attention_control=empty_attention_control,
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
        grid.save("waterfall_cac.png")

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

            # specify the cross-attention control parameters
            contrastive_attention_control_baseline1 = AttentionRefine([prompt, baseline_prompt1],
                                                                      num_inference_steps,
                                                                      cross_replace_steps=1.0,
                                                                      self_replace_steps=0.4,
                                                                      )
            contrastive_attention_control_positive1 = AttentionRefine([prompt, positive_prompt1],
                                                                      num_inference_steps,
                                                                      cross_replace_steps=1.0,
                                                                      self_replace_steps=0.4,
                                                                      )
            contrastive_attention_controls = [
                (
                    lambda pipe: ptp_utils.register_attention_control(pipe, contrastive_attention_control_baseline1),
                    lambda pipe: ptp_utils.register_attention_control(pipe, contrastive_attention_control_positive1)
                ),
            ]
            empty_attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, EmptyControl())

            # call the pipeline
            generator = torch.Generator(device="cuda").manual_seed(0)
            image = pipe(prompt=prompt,
                         contrastive_prompts=contrastive_prompts,
                         contrastive_scales=contrastive_scales,
                         contrastive_attention_controls=contrastive_attention_controls,
                         empty_attention_control=empty_attention_control,
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
        grid.save("valley_cac.png")

    # Continuous, rig-like control
    if False:
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
        for contrastive_scale in [-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]:
            contrastive_scales = [contrastive_scale]

            # specify the cross-attention control parameters
            contrastive_attention_control_baseline1 = AttentionRefine([prompt, baseline_prompt1],
                                                                      num_inference_steps,
                                                                      cross_replace_steps=0.8,
                                                                      self_replace_steps=0.4,
                                                                      )
            contrastive_attention_control_positive1 = AttentionReplace([prompt, positive_prompt1],
                                                                       num_inference_steps,
                                                                       cross_replace_steps=0.8,
                                                                       self_replace_steps=0.4,
                                                                       )
            contrastive_attention_controls = [
                (
                    lambda pipe: ptp_utils.register_attention_control(pipe, contrastive_attention_control_baseline1),
                    lambda pipe: ptp_utils.register_attention_control(pipe, contrastive_attention_control_positive1)
                ),
            ]
            empty_attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, EmptyControl())

            # call the pipeline
            generator = torch.Generator(device="cuda").manual_seed(0)
            image = pipe(prompt=prompt,
                         contrastive_prompts=contrastive_prompts,
                         contrastive_scales=contrastive_scales,
                         contrastive_attention_controls=contrastive_attention_controls,
                         empty_attention_control=empty_attention_control,
                         num_inference_steps=num_inference_steps,
                         guidance_scale=7.5,
                         strength=0.6,
                         generator=generator,
                         ).images[0]
            images.append(image)

        # save the images as a grid
        h, w = images[0].size
        grid = Image.new("RGB", (w * len(images), h))
        for i, image in enumerate(images):
            grid.paste(image, (w * i, 0))
        grid.save("park_cac.png")

    # Continuous, rig-like control
    if False:
        # let's specify a prompt
        prompt = "An oil painting of a couple in formal evening wear going home."

        # let's specify contrastive prompts
        positive_prompt1 = "An oil painting of a couple in formal evening wear going home on a rainy day."
        baseline_prompt1 = "An oil painting of a couple in formal evening wear going home on a sunny day."
        contrastive_prompts = [
            [positive_prompt1, baseline_prompt1],
        ]

        # Enumerate contrastive prompts
        images = []
        for contrastive_scale in [-5, -3.75, -2.5, -1.25, 0, 1.25, 2.5, 3.75, 5]:
            contrastive_scales = [contrastive_scale]

            # specify the cross-attention control parameters
            contrastive_attention_control_baseline1 = AttentionRefine([prompt, baseline_prompt1],
                                                                      num_inference_steps,
                                                                      cross_replace_steps=1.0,
                                                                      self_replace_steps=0.4,
                                                                      )
            contrastive_attention_control_positive1 = AttentionRefine([prompt, positive_prompt1],
                                                                      num_inference_steps,
                                                                      cross_replace_steps=1.0,
                                                                      self_replace_steps=0.4,
                                                                      )
            contrastive_attention_controls = [
                (
                    lambda pipe: ptp_utils.register_attention_control(pipe, contrastive_attention_control_baseline1),
                    lambda pipe: ptp_utils.register_attention_control(pipe, contrastive_attention_control_positive1)
                ),
            ]
            empty_attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, EmptyControl())

            # call the pipeline
            generator = torch.Generator(device="cuda").manual_seed(0)
            image = pipe(prompt=prompt,
                         contrastive_prompts=contrastive_prompts,
                         contrastive_scales=contrastive_scales,
                         contrastive_attention_controls=contrastive_attention_controls,
                         empty_attention_control=empty_attention_control,
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
        grid.save("couple_cac.png")

    # Continuous, rig-like control
    if False:
        # let's specify a prompt
        prompt = "A photo of a fox on a street in new york."

        # let's specify contrastive prompts
        positive_prompt1 = "A overexposure photo of a fox on a street in new york."
        baseline_prompt1 = "A underexposure photo of a fox on a street in new york."
        contrastive_prompts = [
            [positive_prompt1, baseline_prompt1],
        ]

        # Enumerate contrastive prompts
        images = []
        for contrastive_scale in [-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]:
            contrastive_scales = [contrastive_scale]

            # specify the cross-attention control parameters
            contrastive_attention_control_baseline1 = AttentionRefine([prompt, baseline_prompt1],
                                                                      num_inference_steps,
                                                                      cross_replace_steps=1.0,
                                                                      self_replace_steps=0.4,
                                                                      )
            contrastive_attention_control_positive1 = AttentionRefine([prompt, positive_prompt1],
                                                                      num_inference_steps,
                                                                      cross_replace_steps=1.0,
                                                                      self_replace_steps=0.4,
                                                                      )
            contrastive_attention_controls = [
                (
                    lambda pipe: ptp_utils.register_attention_control(pipe, contrastive_attention_control_baseline1),
                    lambda pipe: ptp_utils.register_attention_control(pipe, contrastive_attention_control_positive1)
                ),
            ]
            empty_attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, EmptyControl())

            # call the pipeline
            generator = torch.Generator(device="cuda").manual_seed(0)
            image = pipe(prompt=prompt,
                         contrastive_prompts=contrastive_prompts,
                         contrastive_scales=contrastive_scales,
                         contrastive_attention_controls=contrastive_attention_controls,
                         empty_attention_control=empty_attention_control,
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
        grid.save("brightness_fox_cac.png")

    # Continuous, rig-like control
    if True:
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
        for contrastive_scale in [-10, -5, 0, 5, 10]:
            contrastive_scales = [contrastive_scale]

            # specify the cross-attention control parameters
            contrastive_attention_control_baseline1 = AttentionRefine([prompt, baseline_prompt1],
                                                                      num_inference_steps,
                                                                      cross_replace_steps=1.0,
                                                                      self_replace_steps=0.4,
                                                                      )
            contrastive_attention_control_positive1 = AttentionRefine([prompt, positive_prompt1],
                                                                      num_inference_steps,
                                                                      cross_replace_steps=1.0,
                                                                      self_replace_steps=0.4,
                                                                      )
            contrastive_attention_controls = [
                (
                    lambda pipe: ptp_utils.register_attention_control(pipe, contrastive_attention_control_baseline1),
                    lambda pipe: ptp_utils.register_attention_control(pipe, contrastive_attention_control_positive1)
                ),
            ]
            empty_attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, EmptyControl())

            # call the pipeline
            generator = torch.Generator(device="cuda").manual_seed(0)
            image = pipe(prompt=prompt,
                         contrastive_prompts=contrastive_prompts,
                         contrastive_scales=contrastive_scales,
                         contrastive_attention_controls=contrastive_attention_controls,
                         empty_attention_control=empty_attention_control,
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
        grid.save("banana_cac.png")

    # Continuous, rig-like control
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

            # specify the cross-attention control parameters
            contrastive_attention_control_baseline1 = AttentionRefine([prompt, baseline_prompt1],
                                                                      num_inference_steps,
                                                                      cross_replace_steps=1.0,
                                                                      self_replace_steps=0.4,
                                                                      )
            contrastive_attention_control_positive1 = AttentionRefine([prompt, positive_prompt1],
                                                                      num_inference_steps,
                                                                      cross_replace_steps=1.0,
                                                                      self_replace_steps=0.4,
                                                                      )
            contrastive_attention_controls = [
                (
                    lambda pipe: ptp_utils.register_attention_control(pipe, contrastive_attention_control_baseline1),
                    lambda pipe: ptp_utils.register_attention_control(pipe, contrastive_attention_control_positive1)
                ),
            ]
            empty_attention_control = lambda pipe: ptp_utils.register_attention_control(pipe, EmptyControl())

            # call the pipeline
            generator = torch.Generator(device="cuda").manual_seed(0)
            image = pipe(prompt=prompt,
                         contrastive_prompts=contrastive_prompts,
                         contrastive_scales=contrastive_scales,
                         contrastive_attention_controls=contrastive_attention_controls,
                         empty_attention_control=empty_attention_control,
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
        grid.save("giraffe_cac.png")


if __name__ == "__main__":
    main()
