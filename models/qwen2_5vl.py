import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
import json
import re
import os
from PIL import Image

from qwen_vl_utils import process_vision_info

import re

def parse_string(input_string, image_width, image_height):
    try:
        # Extract the JSON part from input_string using regex
        match = re.search(r'```json\n(.*?)\n```', input_string, re.DOTALL)
        if not match:
            print("Error: JSON data not found in the input string")
            return None
        
        json_data = match.group(1)
        
        # Parse JSON
        data = json.loads(json_data)
        if not data or 'bbox_2d' not in data[0]:
            print("Error: bbox_2d key not found in JSON data")
            return None
        
        x1, y1, x2, y2 = data[0]['bbox_2d']
        
        # Normalize coordinates
        x1_norm = (x1 / image_width) * 1000
        y1_norm = (y1 / image_height) * 1000
        x2_norm = (x2 / image_width) * 1000
        y2_norm = (y2 / image_height) * 1000
        
        # Calculate center coordinates
        return ((x1_norm + x2_norm) / 2, (y1_norm + y2_norm) / 2)
    
    except Exception as e:
        print(f"Error parsing string: {e}")
        return None


def parse_string1(input_string):
    try:
        box = input_string
        return box
    except:
        return input_string

# bbox -> point (str)
def bbox_2_point(bbox, dig=2):
    # bbox [left, top, right, bottom]
    point = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    point = [f"{item:.2f}" for item in point]
    point_str = "({},{})".format(point[0], point[1])
    return point_str

# bbox -> bbox (str)
def bbox_2_bbox(bbox, dig=2):
    bbox = [f"{item:.2f}" for item in bbox]
    bbox_str = "({},{},{},{})".format(bbox[0], bbox[1], bbox[2], bbox[3])
    return bbox_str

# point (str) -> point
def pred_2_point(s):
    floats = re.findall(r'-?\d+\.?\d*', s)
    floats = [float(num) for num in floats]
    if len(floats) == 2:
        return floats
    elif len(floats) == 4:
        return [(floats[0]+floats[2])/2, (floats[1]+floats[3])/2]
    else:
        return None



def extract_bbox(s):
    pattern = r"<\|box_start\|\>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|\>"
    match = re.search(pattern, s)  # Extract first (or last) match

    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        return (x1 + x2, y1 + y2)
    
    return None


def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name

gen_config = GenerationConfig(
)

class Qwen2_5VLModel():
    def load_model(self, model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        # Setting default generation config
        self.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True).to_dict()
        self.set_generation_config(
            max_length=2048,
            do_sample=False,
            temperature=0.0,
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)


    def ground_only_positive(self, instruction, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
            image_width, image_height = image.size  # Get image dimensions
        else:
            image_path = image
            with Image.open(image_path) as img:
                image_width, image_height = img.size  # Get image dimensions

        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."



        prompt_origin = 'Output the bounding box in the image corresponding to the instruction "{}" with grounding. the format should '
        full_prompt = prompt_origin.format(instruction)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]
        # Preparation for inference
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        
        x = image_width
        y = image_height
        response1 = parse_string(response,x,y)
        
        print(response)
        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response1,
            "bbox": None,
            "point": None
        }

        box = parse_string1(response1)
        if type(box) == tuple:
            box = [int(x) / 1000 for x in box] if box is not None else None
        result_dict["point"] = box
        return result_dict


    def ground_allow_negative(self, instruction, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
            image_width, image_height = image.size  # Get image dimensions
        else:
            image_path = image
            with Image.open(image_path) as img:
                image_width, image_height = img.size  # Get image dimensions
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        prompt_origin = 'Output the bounding box in the image corresponding to the instruction "{}". If the target does not exist, respond with "Target does not exist".'
        full_prompt = prompt_origin.format(instruction)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]
        # Preparation for inference
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs,max_new_tokens=1024)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        
        x = image_width
        y = image_height
        response1 = parse_string(response,x,y)

        print(response)

        result_dict = {
            "result": None,
            "format": "x1y1x2y2",
            "raw_response": response1,
            "bbox": None,
            "point": None
        }
        box = parse_string1(response1)
        if type(box) == tuple:
            box = [int(x) / 1000 for x in box] if box is not None else None
        result_dict["point"] = box
        if result_dict["bbox"] or result_dict["point"]:
            result_status = "positive"
        elif "Target does not exist".lower() in response1.lower():
            result_status = "negative"
        else:
            result_status = "wrong_format"
        result_dict["result"] = result_status

        return result_dict
