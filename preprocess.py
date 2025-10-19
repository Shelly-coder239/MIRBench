import json
import os
import re

def replace_image(match):
    """
    Replace image pattern with incremented index.
    Example: Image0 -> Image1
    """
    image_number = int(match.group(1)) + 1
    return f"Image-{image_number}"


def normal_data_transform(item, epoch=4):
    """
    Build structured dialogue messages for multi-stage reasoning data.
    Args:
        item (dict): A single sample entry from MIRBench.
        epoch (int): Indicates the reasoning stage (0–4).
    Returns:
        dict: Formatted message pair for model training.
    """
    messages = []

    # ---- Construct user prompt ----
    if isinstance(item["options"], dict):
        options_text = " ".join([f"{k}: {v}" for k, v in item["options"].items()])
    else:
        options_text = " ".join(item["options"])
    
    user_prompt = f"{item['question']} Select the correct option from the following options: {options_text}"
    assistant_response = ""

    # ---- Stage 0: Full reasoning chain ----
    if epoch == 0:
        assistant_response += (
            item["summary"] + "\n" +
            format_captions(item["caption"]) +
            item["text2img"] + "\n" +
            item["img2img"] + "\n"
        )
        if "reasoning" in item:
            assistant_response += item["reasoning"] + "\n"
        assistant_response += item["conclusion"]

    # ---- Stage 1: Provide all context, predict conclusion ----
    elif epoch == 1:
        user_prompt += build_context(item, include_reasoning=True)
        assistant_response += item["conclusion"]

    # ---- Stage 2: Partial reasoning, model completes later steps ----
    elif epoch == 2:
        user_prompt += build_context(item, up_to="text2img")
        assistant_response += (
            item["img2img"] + "\n" +
            (item.get("reasoning", "") + "\n") +
            item["conclusion"]
        )

    # ---- Stage 3: High-level understanding only ----
    elif epoch == 3:
        user_prompt += build_context(item, up_to="caption")
        assistant_response += (
            item["text2img"] + "\n" +
            item["img2img"] + "\n" +
            (item.get("reasoning", "") + "\n") +
            item["conclusion"]
        )

    # ---- Stage 4: Autonomous reasoning generation ----
    elif epoch == 4:
        assistant_response += (
            item["summary"] + "\n" +
            format_captions(item["caption"], prefix="Let's analyze these images first:") +
            item["text2img"] + "\n" +
            item["img2img"] + "\n"
        )
        if "reasoning" in item:
            assistant_response += item["reasoning"] + "\n"

        if len(item["conclusion"]) > 1:
            assistant_response += item["conclusion"]
        else:
            assistant_response += f"Based on the above, the correct answer is {item['conclusion']}"

    messages.append({"role": "user", "content": user_prompt})
    messages.append({"role": "assistant", "content": assistant_response})
    return {"messages": messages, "images": item["images"]}


def format_captions(captions, prefix=""):
    """
    Helper to format single or multiple image captions.
    """
    text = prefix + ("\n" if prefix else "")
    if isinstance(captions, list):
        for idx, caption in enumerate(captions):
            text += f"Image {idx+1}: {caption}\n"
    else:
        text += captions + "\n"
    return text


def build_context(item, include_reasoning=False, up_to=None):
    """
    Construct progressive reference context for the user prompt.
    up_to: one of [None, 'caption', 'text2img'] to control the amount of context.
    """
    ctx = "\nReference Context:\n"
    ctx += item["summary"] + "\n"
    ctx += format_captions(item["caption"])

    if up_to in (None, "text2img"):
        ctx += item["text2img"] + "\n"
    if up_to in (None, "caption", "text2img"):
        ctx += item["img2img"] + "\n"

    if include_reasoning and "reasoning" in item:
        ctx += item["reasoning"] + "\n"
    return ctx


# Example usage
if __name__ == "__main__":
    with open("/groups/g900403/home/share/zjy/datasets/final_data/train_test_split/hard_train_clean.json", "r") as f:
        data = json.load(f)

    # Test single transformation
    example = normal_data_transform(data[0], epoch=4)
    print(json.dumps(example, indent=2, ensure_ascii=False))
