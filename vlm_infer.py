import os
import re
import json
import torch
from tqdm import tqdm
from PIL import ImageFile
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# Prevent truncated image errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------------
# Configuration
# -------------------------
MODEL_PATH = "../Qwen2.5-VL-7B-Instruct"
DATA_PATH = "../total_valid.json"
OUTPUT_PATH = "vlm_results.json"

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
NUM_SAMPLES = 10  # Generate multiple outputs per input

OPTION_PROMPT = "Please select the most likely answer from the following options:"
POST_PROMPT = "Respond in the following format:\nYour option: ...\nReason: ..."

# -------------------------
# Helper Functions
# -------------------------
def create_messages(question, image_paths, options):
    """
    Construct chat messages for Qwen-VL.
    Handles <image> tags and attaches image metadata.
    """
    text_parts = re.split(r"(<image>)", question)
    messages = [{"role": "user", "content": []}]
    image_index = 0

    # Build message content
    for part in text_parts:
        if part == "<image>" and image_index < len(image_paths):
            messages[0]["content"].append({
                "type": "image",
                "image": image_paths[image_index],
                "max_pixels": MAX_PIXELS,
                "min_pixels": MIN_PIXELS
            })
            image_index += 1
        elif part.strip():
            messages[0]["content"].append({"type": "text", "text": part.strip()})

    # Append options and post-instruction
    options_text = "\nOptions:\n"
    if isinstance(options, list):
        for opt in options:
            options_text += f"{opt}\n"
    else:
        for key, value in options.items():
            options_text += f"{key}: {json.dumps(value)}\n"

    messages[0]["content"].append({
        "type": "text",
        "text": f"{OPTION_PROMPT}\n{options_text.strip()}\n{POST_PROMPT}"
    })
    return messages


def extract_option(response):
    """Extract predicted option letter (A/B/C/...) from model response."""
    match = re.search(r"Option[:\s]*([A-Za-z])", response)
    return match.group(1) if match else "None"


# -------------------------
# Main Inference
# -------------------------
def main():
    # Load dataset
    with open(DATA_PATH, "r") as f:
        data = json.load(f)[12500:]
    print(f"Loaded {len(data)} samples from offset 12500")

    # Initialize model
    llm = LLM(
        model=MODEL_PATH,
        limit_mm_per_prompt={"image": 20},
        max_model_len=32768,
        gpu_memory_utilization=1.7,
        dtype=torch.bfloat16,
        mm_processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
    )

    # Sampling configuration
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1.05,
        max_tokens=512,
        n=NUM_SAMPLES,        # Multiple generations per input
        stop_token_ids=[],
    )

    # Initialize processor
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    results = []
    for idx, item in enumerate(tqdm(data, desc="Running vLLM Inference")):
        try:
            # Construct multi-image message
            messages = create_messages(item["question"], item["images"], item["options"])

            # Apply chat template
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )

            # Build vLLM input
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs

            llm_inputs = {
                "prompt": prompt,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs,
            }

            # Generate multiple responses
            outputs = llm.generate([llm_inputs], sampling_params=sampling_params)

            # Parse multi-sample results
            candidate_texts = [out.text for out in outputs[0].outputs]
            for sample_id, text in enumerate(candidate_texts):
                opt = extract_option(text)
                results.append({
                    "id": idx,
                    "sample_id": sample_id,
                    "option": opt,
                    "response": text
                })

            # Optional: print one example for monitoring
            print(f"\n[{idx}] Generated {len(candidate_texts)} samples. Example:\n{text}\n")

        except Exception as e:
            print(f"[Error @ sample {idx}]: {e}")

    # Save results
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
