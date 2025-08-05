import os
import re
import tempfile
from functools import partial

import pandas as pd

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE, Spatial457_utils
from ..smp import *
from ..utils import track_progress_rich

import yaml
import os
from pathlib import Path

def find_project_root_by_config(config_filename="general_config.yaml"):
    """
    通过查找指定的配置文件，向上遍历目录树来确定项目根目录。

    Args:
        config_filename (str): 项目根目录下的配置文件名，默认为 'general_config.yaml'。

    Returns:
        pathlib.Path: 项目根目录的Path对象。如果找不到，则返回当前工作目录。
    """
    # 从当前文件的绝对路径开始
    current_path = Path(__file__).resolve()

    # 向上遍历所有父目录
    for parent in current_path.parents:
        if (parent / config_filename).exists():
            return parent
    
    # 如果从当前文件路径向上没有找到，尝试从当前工作目录向上遍历
    # 这在某些运行环境下（比如从子目录运行脚本）可能会有用
    current_working_dir = Path(os.getcwd()).resolve()
    for parent in current_working_dir.parents:
        if (parent / config_filename).exists():
            return parent

    # 如果所有尝试都失败了，警告并返回当前工作目录
    print(f"Warning: Project root (marked by '{config_filename}') not found. "
          f"Defaulting to current working directory: {current_working_dir}")
    return current_working_dir
PROJECT_ROOT = find_project_root_by_config()
general_config_yaml_path = PROJECT_ROOT / "general_config.yaml"
with open(general_config_yaml_path, "r") as stream:
    genconf = yaml.safe_load(stream)
DEBUG  = genconf.get("debug", False)
def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
MINI = genconf.get('mini_flag', True)
dprint(f"【DEBUG spatial457.py】mini flag: {MINI}")
#注意，当在Omni3D上测试的时候，无论是用0shot还是其他，都是MINI=True

class Spatial457(ImageBaseDataset):
    TYPE = "VQA"
    # When ROBUST is True, if the models does not follow the format, all of the response will be treated as answers.
    ROBUST = True

    DATASET_URL = {
        "Spatial457": "http://opencompass.openxlab.space/utils/VLMEval/Spatial457.tsv",
        "Spatial457_TEST": "https://huggingface.co/datasets/ysc0034/spatial457_test/raw/main/Spatial457_TEST.tsv",
        "Spatial457_MINI": "https://huggingface.co/datasets/ysc0034/spatial457_700/resolve/main/Spatial457_700.tsv",
        "Spatial457_CLEAN": "https://huggingface.co/datasets/ysc0034/spatial457_clean/resolve/main/Spatial457_CLEAN.tsv",
        #Omni-3D
        "Spatial457_OMNI": "https://huggingface.co/datasets/ysc0034/spatial457_omni/resolve/main/Spatial457_OMNI.tsv"
    }

    DATASET_MD5 = {
        'Spatial457': "1f24f5a7b2cadc3d33a8a66ecf92ca68",
        'Spatial457_TEST': "a7697250ea35a29b8e60463e567a6db0",
        'Spatial457_MINI': "5549f38cc5fabf4eb64970792e3bc35d",
        'Spatial457_CLEAN': "f9fe233b1280b5b94455b53d263ec2d7",
        #Omni-3D
        'Spatial457_OMNI': "68417133da9cc431e2c4f53ea99983c6"
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset_utils = Spatial457_utils()

    def evaluate(self, eval_file, **judge_kwargs):

        data = load(eval_file)
        data['prediction'] = [str(x) for x in data['prediction']]
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]

        all_results = {
            "correct": 0,
            "total": 0,
            "answers": [],
            "format_error": 0,
            
            "L1_single": 0,
            "L2_objects": 0,
            "L3_2d_spatial": 0,
            "L4_occ": 0,
            "L4_pose": 0,
            "L5_6d_spatial": 0,
            "L5_collision": 0,
            
            # Omni-3D 个数统计
            "Omni_3d_str": 0,
            "Omni_3d_int": 0,
            "Omni_3d_float": 0,

            "L1_single_correct": 0,
            "L2_objects_correct": 0,
            "L3_2d_spatial_correct": 0,
            "L4_occ_correct": 0,
            "L4_pose_correct": 0,
            "L5_6d_spatial_correct": 0,
            "L5_collision_correct": 0,
            
            # Omni-3D
            "Omni_3d_str_correct": 0,
            "Omni_3d_int_correct": 0,
            # sum of MRA
            "Omni_3d_float_correct": 0,
        }

        for i in tqdm(range(len(lines))):

            line = lines[i]
            index = int(line["index"])

            answers = str(line["answer"])
            level = line["category"]
            objects = []

            # parse the answer
            # Yang Shucheng 使用原版dataset prompt
            if not MINI:
                #这一部分的触发条件：原版framework + 0-shot wo code Spatial457 (不包含Omni3D)
                pred_try_1 = re.search(r"Answer': '(.*?)'", line["prediction"])
                pred_try_2 = re.search(r'Answer": "(.*?)"', line["prediction"])
                pred_try_3 = re.search(r"Answer': (\d)", line["prediction"])

                if pred_try_1:
                    pred = pred_try_1.group(1)
                elif pred_try_2:
                    pred = pred_try_2.group(1)
                elif pred_try_3:
                    pred = pred_try_3.group(1)
                else:
                    if self.ROBUST:
                        pred = line['prediction']
                    else:
                        pred = self.dataset_utils.get_random_answer(answers)
                    all_results["format_error"] += 1
            

                reasoning_try_1 = re.search(r"Reasoning': '(.*?)'", line["prediction"])
                reasoning_try_2 = re.search(r'Reasoning": "(.*?)"', line["prediction"])

                if reasoning_try_1:
                    reasoning = reasoning_try_1.group(1)
                elif reasoning_try_2:
                    reasoning = reasoning_try_2.group(1)
                else:
                    if self.ROBUST:
                        reasoning = "Format Error. All of the resposne as the answer."
                    else:
                        reasoning = "Format Error. Guess a random answer."
            else:
                #这一部分的触发条件是 : 任何Omni3D + 任何 code mode (sandbox)
                pred = line['prediction']
                reasoning = 'Skipped'
            # YSCE

            #todo 这里如果level里面含有omni,那么要调用omni专用is_correct
            #todo 实现专用is_correct
            if "Omni" in level:
                correct = self.dataset_utils.is_correct_omni(level, answers, pred)
            else:
                correct = self.dataset_utils.is_correct(answers, pred)

            all_results["answers"].append(
                {
                    "index": index,
                    "correct": correct,
                    "answers": answers,
                    "predict": pred,
                    "reasoning": reasoning,
                    "objects": objects,
                }
            )
            #这里要排除Omni_3d_float
            if not ("float" in level):
                all_results["total"] += 1
                if correct:
                    all_results["correct"] += 1
            
            all_results[f"{level}"] += 1
            #这里要分类讨论，如果float则直接加上
            if "float" in level:
                all_results[f"{level}_correct"] += correct
            elif correct:
                all_results[f"{level}_correct"] += 1
        #这里统计的是非float的acc
        all_results["score"] = all_results["correct"] / all_results["total"]

        for level in [
            "L1_single",
            "L2_objects",
            "L3_2d_spatial",
            "L4_occ",
            "L4_pose",
            "L5_6d_spatial",
            "L5_collision",
            #Omni-3D
            "Omni_3d_str",
            "Omni_3d_int",
            "Omni_3d_float",
        ]:
            all_results[f"{level}_score"] = (
                all_results[f"{level}_correct"] / all_results[level] if all_results[level] > 0 else 0
            )

        score_pth = eval_file.replace(".xlsx", "_score.json")

        dump(all_results, score_pth)
        return all_results

    def build_prompt(self, line):
        msgs = super().build_prompt(line)

        set_type = line["category"]
        #Yang Shucheng
        if MINI:
            instruction_1, instruction_2 = self.build_subtask_instruction_ysc_ver(set_type)           
        else:
            instruction_1, instruction_2 = self.build_subtask_instruction(set_type)
        # YSCE
        if instruction_1 != "":
            msgs.insert(0, {"type": "text", "value": instruction_1})
        if instruction_2 != "":
            msgs.append({"type": "text", "value": instruction_2})

        return msgs

    def build_subtask_instruction(self, level):
        if level == 'Omni_3d':
            dprint("【DEBUG spatial457.py】omni 3d dataset prompt activated")
            return "", "Write your response into this json template: " "{'Reasoning': '<your reasons>', 'Answer': '<Your answer>'}"

        task_map = {
            "L1_single": (
                "You are an intelligent chatbot designed to answer questions based on an image. Your task is to "
                "analyze the images, identify attributes of the objects, "
                "and then determine the answer to the question.\n"
            ),
            "L2_objects": (
                "You are an intelligent chatbot designed to answer questions based on an image. Your task is to "
                "analyze the images, identify attributes of multiple objects, "
                "and then determine the answer to the question.\n"
            ),
            "L3_2d_spatial": (
                "You are an intelligent chatbot designed to answer questions based on an image. Your task is to "
                "analyze the images, identify attributes of multiple objects and their spatial relationship from 2D "
                "projected camera view, and then determine the answer to the question.\n"
            ),
            "L4_occ": (
                "You are an intelligent chatbot designed to answer questions based on an image. Your task is to "
                "analyze the images, identify attributes of multiple objects and their occlusion relationships, and "
                "then determine the answer to the question.\n"
            ),
            "L4_pose": (
                "You are an intelligent chatbot designed to answer questions based on an image. Your task is to "
                "analyze the images, identify attributes of multiple objects and their facing direction in 3D space "
                "from the camera view, and then determine the answer to the question.\n"
            ),
            "L5_6d_spatial": (
                "You are an intelligent chatbot designed to answer questions based on an image. Your task is to "
                "analyze the images, identify attributes of multiple objects and their spatial relationship from "
                "objects’ perspective in 3D space, and then determine the answer to the question.\n"
            ),
            "L5_collision": (
                "You are an intelligent chatbot designed to answer questions based on an image. Your task is to "
                "analyze the images, identify attributes of multiple objects and their potential collision given the "
                "assumption of moving direction in 3D space, and then determine the answer to the question.\n"
            ),
        }

        instruction_1 = task_map.get(level, "")

        instruction_2 = (
            "First, you should identify the related objects refered in the questions, including their shape, "
            "color, size; then add a brief reasoning process about the questions. Each object in the image has a "
            "shape (e.g., 'airliner'), a size (only can be 'small' or 'large'), a color (e.g. 'blue'). The size of "
            "the object is either 'small' or 'large'. The color of the object is one of the following: 'gray', "
            "'blue', 'purple', 'brown', 'green', 'cyan', 'red', 'yellow'. The direction of the object is one of the "
            "following: 'left', 'right', 'front', 'back'.\n\n"
            "Second, give the answer based on the reasoning process. The answer should only be (1) a phrase chosen "
            "from the following options: {}, or (2) an integer [0-10] when asked for 'How many' or 'What is the "
            "number of', or (3) 'Yes' or 'No' when asked for 'Is there'. If you think there are no possible answers "
            "or the question is not clear, choose the best answer that fits the question.\n\n"
        ).format(self.dataset_utils.all_answers())

        instruction_2 += (
            "Write your response into this json template: " "{'Reasoning': '<your reasons>', 'Answer': '<Your answer>'}"
        )
        return instruction_1, instruction_2
    
    def build_subtask_instruction_ysc_ver(self, level):
        if 'Omni_3d' in level:
            task_map = {
                "Omni_3d_str": (
                    "Hint:"
                    "If the question asks “Is there …?” or involves comparing counts/attributes, answer must be exactly “yes” or “no”."
                    "If it asks about time, use 24-hour “HH:MM” format (e.g. “12:15”)."
                    "If it asks about direction, like “Which way…?”, choose one from {N, S, E, W, NE, NW, SE, SW}."
                    "If it asks about color, choose one color from {red, blue, green, yellow, black, white, brown, gray}."
                    "If it asks about material, choose one material from {wooden, metal, plastic, glass}."
                    "If it asks about object, name exactly one object or composite object phrase present in the scene (e.g., chair, table, bottle, cup)."
                    "Make sure you answer only the single phrase—no extra words or punctuation."
                ),
                "Omni_3d_int": (
                    "Hint:"
                    "Answer with a non-negative integer count (e.g. 0, 1, 2, …) when the question uses “How many” or “What is the number of”. Do not include any units or additional text—just the integer."
                ),
                "Omni_3d_float": (
                    "Hint:"
                    "Answer with a decimal number (e.g. 2.361) when the question asks for a measurement (e.g. distance, length, angle) or some other float quantity. Do not include units—just the numeric value."
                ) 
            }
            dprint("【DEBUG spatial457.py】omni 3d dataset prompt activated (YSC Ver.)")
            return "", task_map.get(level, "")

        task_map = {
            "L1_single": (
                "Please analyze the images, identify attributes of the objects, "
                "and then determine the answer to the question.\n"
            ),
            "L2_objects": (
                "Please analyze the images, identify attributes of multiple objects, "
                "and then determine the answer to the question.\n"
            ),
            "L3_2d_spatial": (
                "Please analyze the images, identify attributes of multiple objects and their spatial relationship from 2D "
                "projected camera view, and then determine the answer to the question.\n"
            ),
            "L4_occ": (
                "Please analyze the images, identify attributes of multiple objects and their occlusion relationships, and "
                "then determine the answer to the question.\n"
            ),
            "L4_pose": (
                "Please analyze the images, identify attributes of multiple objects and their facing direction in 3D space "
                "from the camera view, and then determine the answer to the question.\n"
            ),
            "L5_6d_spatial": (
                "Please analyze the images, identify attributes of multiple objects and their spatial relationship from "
                "objects’ perspective in 3D space, and then determine the answer to the question.\n"
            ),
            "L5_collision": (
                "Please analyze the images, identify attributes of multiple objects and their potential collision given the "
                "assumption of moving direction in 3D space, and then determine the answer to the question.\n"
            ),
        }

        instruction_1 = task_map.get(level, "")

        instruction_2 = (
            "\nHint 1: Each object in the image has a "
            "shape (e.g., 'airliner'), a size (only can be 'small' or 'large'), a color (e.g. 'blue'). The size of "
            "the object is either 'small' or 'large'. The color of the object is one of the following: 'gray', "
            "'blue', 'purple', 'brown', 'green', 'cyan', 'red', 'yellow'. The direction of the object is one of the "
            "following: 'left', 'right', 'front', 'back'.\n\n"
            "Hint 2: The answer to this question should only be (1) a phrase chosen "
            "from the following options: {}, or (2) an integer [0-10] when asked for 'How many' or 'What is the "
            "number of', or (3) 'Yes' or 'No' when asked for 'Is there'. If you think there are no possible answers "
            "or the question is not clear, choose the best answer that fits the question.\n\n"
        ).format(self.dataset_utils.all_answers())

        return instruction_1, instruction_2
