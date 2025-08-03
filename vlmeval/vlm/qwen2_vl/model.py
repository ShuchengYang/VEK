from __future__ import annotations

import os
import sys
import warnings
import math
import logging

import torch
from transformers import StoppingCriteria

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin
from ...smp import get_gpu_memory, listinstr
from ...dataset import DATASET_MODALITY

# Yang Shucheng Tag
import yaml
import shutil
import requests
import re
import time
import torch.distributed as dist 
sample_counter=0
LRZ_MODE = True
from pathlib import Path
# 根据 model.py 的位置，向上回溯到项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[3]
general_config_yaml_path = PROJECT_ROOT / "general_config.yaml"
with open(general_config_yaml_path, "r") as stream:
    genconf = yaml.safe_load(stream)


# please check
DEBUG  = genconf.get("debug", False)
USE_ORIGIN = genconf.get("use_origin", False)
TEMP_IMG_FOLDER = genconf.get('sandbox_temp_img_folder', "./delete-me-sandbox-temp-img-folder") # sandbox side
LOCAL_SMALL = genconf.get('local_temp_img_folder', "delete-me-local-temp-img-folder")
LOCAL_TEMP_IMG_FOLDER = PROJECT_ROOT / LOCAL_SMALL
def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
dprint(f"【DEBUG model.py】debug mode: {DEBUG}")
dprint(f"【DEBUG model.py】local temp img folder: {LOCAL_TEMP_IMG_FOLDER}")
dprint(f"【DEBUG model.py】sandbox temp img folder: {TEMP_IMG_FOLDER}")
# YSCE

VLLM_MAX_IMAGE_INPUT_NUM = 24


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def create_image_content(image_path, min_pixels, max_pixels):
    base64_image, mime_type = encode_image(image_path)
    return {
        "type": "image",
        "image": f"data:{mime_type};base64,{base64_image}",
        'min_pixels': min_pixels,
        'max_pixels': max_pixels
    }


def encode_image(image_path, max_side=None):
    from mimetypes import guess_type
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    image_format = mime_type.split("/")[-1].upper() if mime_type else "JPEG"

    from PIL import Image
    image = Image.open(image_path)
    # Handle the alpha channel
    if image.mode == "RGBA":
        image = _rgba_to_rgb(image)
    if max_side:
        image = _resize_image(image, max_side)
    encoded_image = _encode_image(image, image_format)

    return encoded_image, mime_type


def _encode_image(image, image_format):
    from io import BytesIO
    with BytesIO() as output:
        image.convert("RGB").save(output, format=image_format)
        import base64
        base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
    return base64_encoded_data


def _rgba_to_rgb(image):
    from PIL import Image
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image).convert("RGB")


def _resize_image(image, max_side):
    resize_scale = max_side / max(image.size)
    new_size = (
        int(image.size[0] * resize_scale),
        int(image.size[1] * resize_scale),
    )
    return image.resize(new_size)


def process_video(video_path, num_frames, min_pixels, max_pixels):
    import cv2
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

    # the sampling rate using max number of frames
    sampling_gap_maxframe = (
        1 if not num_frames else math.ceil(frame_count / num_frames)
    )
    sampling_gap = max(math.ceil(fps / 5), sampling_gap_maxframe)

    frame_number = 0
    images = []

    while True:
        import tempfile
        success, frame = cap.read()
        if not success:
            break
        # Sample frames based on the dynamic sampling rate
        if frame_number % sampling_gap == 0:
            # Create a temporary file for the frame
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as temp_frame:
                cv2.imwrite(temp_frame.name, frame)
                images.append(create_image_content(temp_frame.name, min_pixels, max_pixels))
                os.remove(temp_frame.name)
        frame_number += 1
    if frame_number == 0:
        raise ValueError(f"Failed to read video from {video_path}, check data...")
    logging.info(
        f"Sampled {len(images)}/{frame_number} frames from video {video_path}"
    )
    cap.release()
    return images


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if (
                len(cur_keyword_ids) > 1
                and cur_keyword_ids[0] == tokenizer.bos_token_id
            ):
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [
            keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids
        ]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(
            output_ids[:, -offset:], skip_special_tokens=True
        )[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False


CHAT_TEMPLATE = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"  # noqa: E501

UNTIL = ["<|diff_marker|>"]

class Qwen2VLChat(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,  # if True, will try to only extract stuff in the last \boxed{}.
        verbose: bool = False,
        use_audio_in_video: bool = False,
        # Yang Shucheng Tag
        code_mode : bool = True,
        # YSCE
        **kwargs,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        if self.total_pixels and self.total_pixels > 24576 * 28 * 28:
            print('The total number of video tokens might become too large, resulting in an overly long input sequence. We recommend lowering **total_pixels** to below **24576 × 28 × 28**.')  # noqa: E501
        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)
        if self.fps is None and self.nframe is None:
            print("Warning: fps and nframe are both None, \
                  using default nframe/fps setting in qwen-vl-utils/qwen-omni-utils, \
                  the fps/nframe setting in video dataset is omitted")
        self.use_audio_in_video = use_audio_in_video
        self.FRAME_FACTOR = 2
        assert model_path is not None
        self.model_path = model_path
        MODEL_CLS = None

        if listinstr(['omni'], model_path.lower()):
            try:
                from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
            except Exception as err:
                logging.critical("pip install git+https://github.com/huggingface/transformers@3a1ead0aabed473eafe527915eea8c197d424356")  # noqa: E501
                raise err
            MODEL_CLS = Qwen2_5OmniForConditionalGeneration
            self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        # Yang Shucheng Tag
        elif listinstr(['qwen2_'], model_path.lower()):
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
            MODEL_CLS = Qwen2VLForConditionalGeneration
            self.processor = Qwen2VLProcessor.from_pretrained(model_path)
        elif listinstr(['ysc', '2.5', '2_5', 'qwen25', 'mimo'], model_path.lower()):
        # YSCE
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
            MODEL_CLS = Qwen2VLForConditionalGeneration
            self.processor = Qwen2VLProcessor.from_pretrained(model_path)

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        # assert max_gpu_mem > 0
        self.use_vllm = kwargs.get('use_vllm', False)
        self.use_lmdeploy = kwargs.get('use_lmdeploy', False)
        self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM
        assert self.use_vllm + self.use_lmdeploy <= 1, "You can only set one flag between `use_vllm` and `use_lmdeploy` to True"  # noqa: E501

        if self.use_vllm:
            from vllm import LLM
            gpu_count = torch.cuda.device_count()
            if gpu_count >= 8:
                tp_size = 8
            elif gpu_count >= 4:
                tp_size = 4
            elif gpu_count >= 2:
                tp_size = 2
            else:
                tp_size = 1
            logging.info(
                f'Using vLLM for {self.model_path} inference with {tp_size} GPUs (available: {gpu_count})'
            )
            import os
            if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
                logging.warning(
                    'VLLM_WORKER_MULTIPROC_METHOD is not set to spawn.'
                    'Use \'export VLLM_WORKER_MULTIPROC_METHOD=spawn\' to avoid potential multi-process issues'
                )
            self.llm = LLM(
                model=self.model_path,
                max_num_seqs=5,
                max_model_len=32768,
                limit_mm_per_prompt={"image": self.limit_mm_per_prompt},
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
            )

        elif self.use_lmdeploy:
            from lmdeploy import TurbomindEngineConfig, pipeline, ChatTemplateConfig
            num_gpus = torch.cuda.device_count()
            self.model = pipeline(
                model_path,
                backend_config=TurbomindEngineConfig(session_len=32768, cache_max_entry_count=0.1, tp=num_gpus),
                chat_template_config=ChatTemplateConfig(model_name='qwen2d5-vl'))
            torch.cuda.set_device(0)
            self.device = 'cuda'
        else:
            # self.model = MODEL_CLS.from_pretrained(
            #     model_path, torch_dtype='auto', device_map="auto", attn_implementation='flash_attention_2'
            # )
            # Yang Shucheng Tag
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype='auto', device_map="auto", attn_implementation='sdpa'
            )
            # YSCE
            self.model.eval()
        
        # Yang Shucheng Tag
        self.code_mode = code_mode
        self.sandbox_url = 'http://10.153.51.195:8080/api/sandbox/execute'
        self.prompt_template_yaml_path = PROJECT_ROOT / 'yangshucheng-compass' / 'prompt_template.yaml'
        
        with open(self.prompt_template_yaml_path, "r") as stream:
                conf = yaml.safe_load(stream)
        
        self.active_tools, self.filtered_meta = self.load_tool_data(conf)
        self.tool_list = ", ".join(self.active_tools)
        self.prompt_template = conf.get("prompt_template", {})
        self.code_prompt_template = conf.get("code_prompt_template", {})
        self.temp_image_folder = TEMP_IMG_FOLDER
        self.local_temp_image_folder = LOCAL_TEMP_IMG_FOLDER
        # YSCE
        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
            elif s['type'] == 'video':
                item = {
                    'type': 'video',
                    'video': ensure_video_url(s['value'])
                }
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            elif s['type'] == 'audio':
                item = {'type':'audio','audio':s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def _prepare_content_vllm(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        video_inputs = [s for s in inputs if s['type'] == 'video']
        video_count = len(video_inputs)
        cur_image_count = 0
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                if cur_image_count < self.limit_mm_per_prompt:
                    content.append(item)
                    cur_image_count += 1
                else:
                    logging.warning(
                        f"Number of images exceeds the limit of {self.limit_mm_per_prompt}. "
                        f"Only the first {self.limit_mm_per_prompt} images will be used."
                    )
            elif s['type'] == 'video':
                if video_count > 1:
                    logging.warning(
                        "Multiple videos detected. Using video frames for each video"
                    )
                    if dataset == 'OCRBench':
                        min_pixels = 10 * 10 * 28 * 28
                        warnings.warn(f"OCRBench dataset uses custom min_pixels={min_pixels}")
                        if self.max_pixels is not None:
                            max_pixels = self.max_pixels
                    else:
                        if self.min_pixels is not None:
                            min_pixels = self.min_pixels
                        if self.max_pixels is not None:
                            max_pixels = self.max_pixels
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()

                    frames_per_video = max(1, self.limit_mm_per_prompt // video_count)
                    content.append({"type": "text", "text": "<video frames start>"})
                    content.extend(process_video(s['value'], frames_per_video, min_pixels, max_pixels))
                    content.append({"type": "text", "text": "<video frames end>"})

                else:
                    item = {
                        'type': 'video',
                        'video': ensure_video_url(s['value'])
                    }
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                    if self.total_pixels is not None:
                        item['total_pixels'] = self.total_pixels
                    if self.fps is not None:
                        item['fps'] = self.fps
                    elif self.nframe is not None:
                        import cv2
                        video = cv2.VideoCapture(s['value'])
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()
                        if frame_count < self.nframe:
                            new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                            print(f"use {new_frame_count} for {s['value']}")
                            item['nframes'] = new_frame_count
                        else:
                            item['nframes'] = self.nframe
                    content.append(item)
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
                content.append(item)
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
        return content

    def generate_inner_transformers(self, message, dataset=None):
        if listinstr(['omni'], self.model_path.lower()):
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, please install it via 'pip install qwen-omni-utils[decord]'")  # noqa: E501
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")  # noqa: E501
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        if listinstr(['omni'], self.model_path.lower()):
            audios, images, videos = process_mm_info([messages], use_audio_in_video=self.use_audio_in_video)
            inputs = self.processor(text=text, images=images,audio=audios, videos=videos, padding=True, return_tensors='pt',use_audio_in_video=self.use_audio_in_video)  # noqa: E501
        else:
            images, videos = process_vision_info([messages])
            inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')  # noqa: E501
        inputs = inputs.to('cuda')

        if listinstr(['omni'], self.model_path.lower()):
            self.generate_kwargs['use_audio_in_video'] = self.use_audio_in_video
            self.generate_kwargs['return_audio'] = False
        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response

    def generate_inner_lmdeploy(self, message, dataset=None):
        from lmdeploy import GenerationConfig
        gen_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            top_p=self.generate_kwargs['top_p'],
            top_k=self.generate_kwargs['top_k'],
            temperature=self.generate_kwargs['temperature'],
            repetition_penalty=self.generate_kwargs['repetition_penalty'],
        )
        gen_config.random_seed = None
        messages_list = self.message_to_lmdeploy(message, system_prompt=self.system_prompt)
        assert len(messages_list) == 1
        response = self.model(messages_list, gen_config=gen_config)[0]
        response = response.text
        return response

    def generate_inner_vllm(self, message, dataset=None):
        from vllm import SamplingParams

        if listinstr(['omni'], self.model_path.lower()):
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, please install it via 'pip install qwen-omni-utils[decord]'")  # noqa: E501
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")  # noqa: E501
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content_vllm(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if listinstr(['omni'], self.model_path.lower()):
            audios, images, videos = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
        else:
            images, videos = process_vision_info(messages)
        print('finishing process vision info in vllm.')

        if DATASET_MODALITY(dataset) == 'VIDEO' and 'megabench' not in dataset.lower():
            assert len(videos) == 1
            videos_nd = [videos[0].detach().cpu().numpy().transpose(0, 2, 3, 1)]

            video_inputs = {
                "prompt": text[0],
                "multi_modal_data": {"video": videos_nd[0]},
                "mm_processor_kwargs":{}
            }
            if self.use_audio_in_video:
                import vllm
                assert not vllm.envs.VLLM_USE_V1, ("V1 does not support use_audio_in_video. Please launch this example with `VLLM_USE_V1=0`.")  # noqa: E501
                video_inputs["multi_modal_data"]["audio"] = audios[0]
                video_inputs['mm_processor_kwargs']['use_audio_in_video'] = True
            if videos_nd[0].shape[0] > VLLM_MAX_IMAGE_INPUT_NUM:
                print('video input sequence may be too long for vllm, Maybe cannot generate response for VLLM')
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=self.max_new_tokens, stop_token_ids=None
        )
        if images:
            outputs = self.llm.generate(
                {
                    "prompt": text,
                    "multi_modal_data": {"image": images},
                },
                sampling_params=sampling_params,
            )
        elif videos_nd:
            outputs = self.llm.generate(
                video_inputs,
                sampling_params=sampling_params,
            )
        else:
            outputs = self.llm.generate(
                {
                    "prompt": text,
                },
                sampling_params=sampling_params,
            )

        for o in outputs:
            generated_text = o.outputs[0].text

        if self.post_process:
            resp = generated_text.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                generated_text = resp[:end]

        if self.verbose:
            print(f'\033[32m{generated_text}\033[0m')
        return generated_text

    # Yang Shucheng Tag
    # load selected tooldata from prompt yaml file        
    def load_tool_data(self, conf):
        # --- Tool Metadata Filtering Logic ---
        active_tool_names = conf.get("available_tools", []) # Get the list from YAML
        full_toolbox_metadata = conf.get("toolbox_metadata", {})

        # Create a dictionary containing only the metadata for active tools
        filtered_metadata_dict = {
            tool_name: full_toolbox_metadata[tool_name]
            for tool_name in active_tool_names
            if tool_name in full_toolbox_metadata
        }

        # Warn for missing tools
        for tool_name in active_tool_names:
            if tool_name not in full_toolbox_metadata:
                print(f"Warning: Tool '{tool_name}' listed in available_tools but not found in toolbox_metadata.")

        return active_tool_names, filtered_metadata_dict
    
    # Yang Shucheng Tag
    def _to_sandbox(self, response, image_payload):
        dprint(f"\n【DEBUG/】response:\n{response}\n【/DEBUG】\n")
        # todo 如果检测到到response含有<answer></answer>，则直接提取中间的内容作为response
        # todo 如果检测到到response含有<code></code>，则提取中间的内容，发送给sandbox执行，然后返回
        # 如果sandbox返回结果字典中result的值为None，则直接返回"None"字样
        # 如果sandbox返回结果字典中result的值不为None，则直接返回result的值作为response    
        # todo 如果检测到response含有<answer></answer>，则直接提取中间的内容作为response
        answer_found = False
        m = re.search(r'<answer>(.*?)</answer>', response, re.S)
        if m:
            answer_found = True
            response = m.group(1).strip()

        code_found = False
        if not answer_found:
            # todo 如果检测到response含有<code></code>，则提取中间的内容，发送给sandbox执行，然后返回
            m = re.search(r'<code>(.*?)</code>', response, re.S)
            if m:
                code_found = True
                code_to_exec = m.group(1)
                payload = {
                    "code": code_to_exec,
                    "timeout": 300,
                    "q_aid": f"qa_{int(time.time())}",
                    "images":  image_payload
                }
                try:
                    resp = requests.post(self.sandbox_url, json=payload, timeout=payload.get('timeout', 300)+15)
                    result_data = resp.json()
                    exec_result = result_data.get('result')
                    response = "None" if exec_result is None else exec_result
                except Exception:
                    response = "[FAILED TO GENERATE ANSWER, SANDBOX SERVICE ERROR]"
        if not answer_found and not code_found:
            response = "[FAILED TO GENERATE ANSWER]"
        dprint(f"\n【DEBUG model.py】prediction: {response}")
        return response
    
    def _construct_conversation(self, message, dataset=None, vllm_flag=False):
        # some prepare work before cleaning folder
        try:
            rank = dist.get_rank()
        except Exception:
            rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        # sandbox side
        rank_folder = os.path.join(self.temp_image_folder, f"rank_{rank}")
        #to download img
        local_rank_folder = os.path.join(self.local_temp_image_folder, f"rank_{rank}")

        # 这里要用自定义的模板构造message
        messages = []
        if vllm_flag:
            content = self._prepare_content_vllm(message, dataset=dataset)
        else:
            content = self._prepare_content(message, dataset=dataset)
        # content is a list of dicts, each dict has keys: ['type', 'value']
        
        # make conversation here
        question_chunks = []
        
        # for prompt construction
        image_paths = []
        # img as payload, should use sandbox side path
        image_payload = []

        # 清空文件夹 self.local_temp_image_folder
        if os.path.exists(local_rank_folder):
            shutil.rmtree(local_rank_folder)
        os.makedirs(local_rank_folder, exist_ok=True)
        
        final_content = []
        img_idx = 0

        for item in content:
            if item['type'] == 'text':
                question_chunks.append(item['text'])
                # dprint(f"【DEBUG model.py】question_chunk: {item['text']}")
            elif item['type'] == 'image':
                # 下载 / 复制到本地
                img_src = item['image']
                if img_src.startswith('file://') or not img_src.startswith(('http://','https://')):
                    src_path = img_src.replace('file://','')
                    if not os.path.isabs(src_path):
                        src_path = os.path.abspath(src_path)
                    base_name = f"idx_{img_idx}_{os.path.basename(src_path)}"
                    img_idx += 1
                    local_path = os.path.join(local_rank_folder, base_name)
                    sandbox_path = os.path.join(rank_folder, base_name)
                    shutil.copy(src_path, local_path)
                else:
                    resp = requests.get(img_src, timeout=30)
                    base_name = f"idx_{img_idx}_{os.path.basename(img_src)}"
                    img_idx += 1
                    local_path = os.path.join(local_rank_folder, base_name)
                    sandbox_path = os.path.join(rank_folder, base_name)
                    with open(local_path,'wb') as f:
                        f.write(resp.content)

                # 1) 收集路径用于 prompt
                image_paths.append(sandbox_path)
                dprint(f"【DEBUG{img_idx} model.py】local path: {local_path}")
                dprint(f"【DEBUG{img_idx} model.py】sandbox_path: {sandbox_path}")

                # 2) 对本地文件做 Base64 编码，构造 payload
                encoded, _mime = encode_image(local_path)   # 使用你已有的 encode_image 辅助
                image_payload.append({
                    "path": sandbox_path,
                    "data": encoded
                })

                # 3) 原样保留 item 以便 prompt 构造
                final_content.append(item)
            elif item['type'] == 'video' or item['type'] == 'audio':
                final_content.append(item)
            else:
                raise ValueError(f"Invalid message type: {item['type']}, {item}")
        question = " ".join(question_chunks)
        # replace template
        keywords = {
            "question": question,
            "image_paths": ", ".join(image_paths),
            "available_tools": self.tool_list,
            "toolbox_metadata": self.filtered_meta,
        }
        dprint(f"\n【DEBUG/】question:\n{keywords['question']}\n【/DEBUG】\n")
        dprint(f"\n【DEBUG/】image_paths(sandbox):\n{keywords['image_paths']}\n【/DEBUG】\n")
        if self.code_mode:
            formatted = self.code_prompt_template.format(**keywords)
        else:
            formatted = self.prompt_template.format(**keywords)
        # dprint(f"\n【DEBUG/】formatted:\n{formatted}\n【/DEBUG】\n")
        final_content.append({
            "type": "text",
            "text": formatted
        })
        
        #define our own system prompt
        if self.code_mode:
            sys_prompt = (
                "You are an expert AI assistant that generates Python code to solve problems using a given set of tools. Your task is to write a Python script that answers a user's question."
                "Your response MUST STRICTLY be a single Python code block enclosed in `<code>` and `</code>` tags."
                )
            messages.append({'role': 'system', 'content': sys_prompt})
        messages.append({'role': 'user', 'content': final_content})
        # end of conversation construction
        return messages, image_payload
    
    def generate_inner_yangshucheng(self, message, dataset=None):
        if listinstr(['omni'], self.model_path.lower()):
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, please install it via 'pip install qwen-omni-utils[decord]'")  # noqa: E501
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")  # noqa: E501
                raise err

        # Yang Shucheng some prepare work before cleaning folder
        messages, image_payload = self._construct_conversation(message, dataset, vllm_flag=False)
        # end of conversation construction 
        
        # keep the way it is
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        if listinstr(['omni'], self.model_path.lower()):
            audios, images, videos = process_mm_info([messages], use_audio_in_video=self.use_audio_in_video)
            inputs = self.processor(text=text, images=images,audio=audios, videos=videos, padding=True, return_tensors='pt',use_audio_in_video=self.use_audio_in_video)  # noqa: E501
        else:
            images, videos = process_vision_info([messages])
            inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')  # noqa: E501
        inputs = inputs.to('cuda')

        if listinstr(['omni'], self.model_path.lower()):
            self.generate_kwargs['use_audio_in_video'] = self.use_audio_in_video
            self.generate_kwargs['return_audio'] = False
        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]

        # Yang Shucheng,  to sandbox
        response = self._to_sandbox(response, image_payload)
        # YSCE

        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response

    def generate_inner_vllm_yangshucheng(self, message, dataset=None):
        from vllm import SamplingParams

        if listinstr(['omni'], self.model_path.lower()):
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, please install it via 'pip install qwen-omni-utils[decord]'")  # noqa: E501
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")  # noqa: E501
                raise err
        
        # Yangshucheng
        dprint("vllm mode conversation construction")
        messages, image_payload = self._construct_conversation(message, dataset, True)
        # YSCE
        
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if listinstr(['omni'], self.model_path.lower()):
            audios, images, videos = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
        else:
            images, videos = process_vision_info(messages)
        print('finishing process vision info in vllm.')

        if DATASET_MODALITY(dataset) == 'VIDEO' and 'megabench' not in dataset.lower():
            assert len(videos) == 1
            videos_nd = [videos[0].detach().cpu().numpy().transpose(0, 2, 3, 1)]

            video_inputs = {
                "prompt": text[0],
                "multi_modal_data": {"video": videos_nd[0]},
                "mm_processor_kwargs":{}
            }
            if self.use_audio_in_video:
                import vllm
                assert not vllm.envs.VLLM_USE_V1, ("V1 does not support use_audio_in_video. Please launch this example with `VLLM_USE_V1=0`.")  # noqa: E501
                video_inputs["multi_modal_data"]["audio"] = audios[0]
                video_inputs['mm_processor_kwargs']['use_audio_in_video'] = True
            if videos_nd[0].shape[0] > VLLM_MAX_IMAGE_INPUT_NUM:
                print('video input sequence may be too long for vllm, Maybe cannot generate response for VLLM')
        
        # tokenizer=self.processor.tokenizer
        # eos_id= tokenizer.eos_token_id 
        sampling_params = SamplingParams(
            # temperature=self.generate_kwargs['temperature'],
            temperature=0.5,
            # top_p=self.generate_kwargs['top_p'],
            # top_k=self.generate_kwargs['top_k'],
            max_tokens=self.max_new_tokens, 
            # stop=['</code>','</answer>'],
            # stop_token_ids=[eos_id],
            # include_stop_str_in_output=True,
            # repetition_penalty=self.generate_kwargs['repetition_penalty'],
        )
        # sampling_params = SamplingParams(
        #     # —————————————————————————————————————————————————————————————
        #     # 随机度控制（略微探索，避免完全贪心重复）
        #     temperature=0.1,         # 0.0→纯贪心；0.2→保留少量随机性
        #     top_p=0.85,              # nucleus sampling，累积概率阈值
        #     top_k=5,                # 从概率最高的 40 个 token 里采样
        #     # —————————————————————————————————————————————————————————————
        #     # 重复惩罚（抑制整段复读）
        #     repetition_penalty=1.1,  # >1.0 更不愿意重复同一 token
        #     # frequency_penalty=0.5,   # 根据出现次数惩罚高频 token
        #     # presence_penalty=0.5,    # 只要出现过就惩罚，鼓励使用新 token
        #     # —————————————————————————————————————————————————————————————
        #     # 停机条件（只在遇到自定义标签时停机）
        #     stop=["</code>", "</answer>"],
        #     # stop=['<|im_end|>'],
        #     include_stop_str_in_output=True,
        #     # 如果你还是想用 EOS 作为唯一停机点，可以去掉 stop_sequences，改成：
        #     # stop_token_ids=[eos_id],
        #     # —————————————————————————————————————————————————————————————
        #     max_tokens=self.max_new_tokens,    # 与 Transformers 同步你的 max_new_tokens
        # )
        dprint(f"【DEBUG\】text:\n{text}【\DEBUG】")
        if images:
            outputs = self.llm.generate(
                {
                    "prompt": text,
                    "multi_modal_data": {"image": images},
                },
                sampling_params=sampling_params,
            )
        elif videos_nd:
            outputs = self.llm.generate(
                video_inputs,
                sampling_params=sampling_params,
            )
        else:
            outputs = self.llm.generate(
                {
                    "prompt": text,
                },
                sampling_params=sampling_params,
            )

        for o in outputs:
            generated_text = o.outputs[0].text
        
        # Yang Shucheng
        # dprint("vllm mode to sandbox")
        generated_text = self._to_sandbox(generated_text, image_payload)
        # YSCE

        if self.post_process:
            resp = generated_text.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                generated_text = resp[:end]

        if self.verbose:
            print(f'\033[32m{generated_text}\033[0m')
        return generated_text

    # YSCE  
    # Yang Shucheng Tag
    def generate_inner(self, message, dataset=None):
        global sample_counter
        sample_counter += 1
        # guarding dist.barrier() with dist.is_initialized() is a perfectly valid way 
        # to make sure you only do the sync when you’ve actually called init_process_group.
        
        # if there’s any chance this code might run in a single‑GPU or 
        # non‑distributed context (i.e. you never did init_process_group) 
        # then calling dist.barrier() directly will raise an exception.
        
        # You just need to make sure each rank’s sample_counter starts 
        # from the same value (e.g. 0) and 
        # that every rank calls generate_inner the same number of times.
        if dist.is_initialized() and sample_counter % 10 == 0:
            dist.barrier()
        if self.use_vllm:
            if USE_ORIGIN:
                return self.generate_inner_vllm(message, dataset=dataset)
            return self.generate_inner_vllm_yangshucheng(message, dataset=dataset)
        else:
            return self.generate_inner_yangshucheng(message, dataset=dataset)
        
        # if self.use_vllm:
        #     return self.generate_inner_vllm(message, dataset=dataset)
        # elif self.use_lmdeploy:
        #     return self.generate_inner_lmdeploy(message, dataset=dataset)
        # else:
        #     return self.generate_inner_transformers(message, dataset=dataset)


class Qwen2VLChatAguvis(Qwen2VLChat):
    def __init__(self, mode=None, **kwargs):
        self.mode = mode
        super().__init__(**kwargs)
        self.processor.max_pixels = self.max_pixels
        self.processor.min_pixels = self.min_pixels

    def generate_inner(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical(
                "qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'"
            )
            raise err

        messages = []
        user_message = []
        for item in message:
            if "role" in item.keys():
                if item["role"] == "system":
                    self.system_prompt = item["value"]
                else:
                    item.pop("role")
                    user_message.append(item)
            else:
                user_message.append(item)
        message = user_message

        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {"role": "user", "content": self._prepare_content(message, dataset=dataset)}
        )
        if self.verbose:
            print(f"\033[31m{messages}\033[0m")

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            chat_template=CHAT_TEMPLATE,
        )
        # TODO: provide current action's low-level instruction
        # if False:
        #     # If low-level instruction is provided
        #     # We enforce using "Action: {low_level_instruction} to guide generation"
        #     recipient_text = f"<|im_start|>assistant<|recipient|>all\nAction: {low_level_instruction}\n"
        if self.mode == "force-plan":
            recipient_text = "<|im_start|>assistant<|recipient|>all\nThought: "
        elif self.mode == "force-plan-l1":
            recipient_text = "<|im_start|>assistant<|recipient|>all\nAction: "
        elif self.mode == "force-plan-l3":
            recipient_text = "<|im_start|>assistant<|recipient|>all\nObservation: "
        elif self.mode == "grounding":
            recipient_text = "<|im_start|>assistant<|recipient|>os\n"
        elif self.mode == "force-plan-free":
            recipient_text = "<|im_start|>assistant<|recipient|>all\n"
        elif self.mode == "self-plan":
            recipient_text = "<|im_start|>assistant<|recipient|>"
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        text += recipient_text
        # print(text)

        images, videos = process_vision_info([messages])
        inputs = self.processor(
            text=[text], images=images, videos=videos, padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        # stop_str = "<|diff_marker|>"
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(
        #     keywords, self.processor.tokenizer, inputs.input_ids
        # )

        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
            # stopping_criteria=[stopping_criteria],
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        # for term in UNTIL:
        #     if len(term) > 0:
        #         response = response.split(term)[0]

        if self.post_process:
            resp = response.split("\\boxed{")[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == "{":
                    counter += 1
                elif resp[i] == "}":
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f"\033[32m{response}\033[0m")
        return response
