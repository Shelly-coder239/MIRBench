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


def validate_input_item(item):
    """
    Validate input item structure and required fields.
    Args:
        item (dict): Input data item
    Returns:
        bool: True if valid, False otherwise
    """
    required_fields = ['question', 'options', 'summary', 'caption', 
                      'text2img', 'img2img', 'conclusion']
    
    if not isinstance(item, dict):
        return False
    
    for field in required_fields:
        if field not in item:
            return False
    
    return True


def normal_data_transform(item, epoch=4):
    """
    Build structured dialogue messages for multi-stage reasoning data.
    Args:
        item (dict): A single sample entry from MIRBench.
        epoch (int): Indicates the reasoning stage (0–4).
    Returns:
        dict: Formatted message pair for model training.
    """
    # Validate input
    if not validate_input_item(item):
        raise ValueError("Invalid input item: missing required fields or incorrect structure")
    
    # Validate epoch range
    if not 0 <= epoch <= 4:
        raise ValueError(f"Epoch must be between 0 and 4, got {epoch}")
    
    messages = []

    # ---- Construct user prompt ----
    if isinstance(item["options"], dict):
        options_text = " ".join([f"{k}: {v}" for k, v in item["options"].items()])
    else:
        options_text = " ".join(item["options"])
    
    user_prompt = f"{item['question']} Select the correct option from the following options: {options_text}"
    
    # Use list for more efficient string concatenation
    assistant_response_parts = []

    # ---- Stage 0: Full reasoning chain ----
    if epoch == 0:
        assistant_response_parts.extend([
            item["summary"], "\n",
            format_captions(item["caption"]),
            item["text2img"], "\n",
            item["img2img"], "\n"
        ])
        if "reasoning" in item:
            assistant_response_parts.extend([item["reasoning"], "\n"])
        assistant_response_parts.append(item["conclusion"])

    # ---- Stage 1: Provide all context, predict conclusion ----
    elif epoch == 1:
        user_prompt += build_context(item, include_reasoning=True)
        assistant_response_parts.append(item["conclusion"])

    # ---- Stage 2: Partial reasoning, model completes later steps ----
    elif epoch == 2:
        user_prompt += build_context(item, up_to="text2img")
        assistant_response_parts.extend([
            item["img2img"], "\n",
            item.get("reasoning", ""), "\n",
            item["conclusion"]
        ])

    # ---- Stage 3: High-level understanding only ----
    elif epoch == 3:
        user_prompt += build_context(item, up_to="caption")
        assistant_response_parts.extend([
            item["text2img"], "\n",
            item["img2img"], "\n",
            item.get("reasoning", ""), "\n",
            item["conclusion"]
        ])

    # ---- Stage 4: Autonomous reasoning generation ----
    elif epoch == 4:
        assistant_response_parts.extend([
            item["summary"], "\n",
            format_captions(item["caption"], prefix="Let's analyze these images first:"),
            item["text2img"], "\n",
            item["img2img"], "\n"
        ])
        if "reasoning" in item:
            assistant_response_parts.extend([item["reasoning"], "\n"])

        if len(item["conclusion"]) > 1:
            assistant_response_parts.append(item["conclusion"])
        else:
            assistant_response_parts.append(f"Based on the above, the correct answer is {item['conclusion']}")

    # Join all parts for final response
    assistant_response = "".join(assistant_response_parts)
    
    messages.append({"role": "user", "content": user_prompt})
    messages.append({"role": "assistant", "content": assistant_response})
    
    # Handle optional 'images' field
    images = item.get("images", None)
    
    return {"messages": messages, "images": images}



def format_captions(captions, prefix=""):
    """
    Helper to format single or multiple image captions.
    Args:
        captions: Single caption string or list of captions
        prefix: Optional prefix text
    Returns:
        str: Formatted caption text
    """
    # Use list for more efficient string concatenation
    parts = []
    if prefix:
        parts.extend([prefix, "\n"])
    
    if isinstance(captions, list):
        for idx, caption in enumerate(captions):
            parts.extend([f"Image {idx+1}: {caption}", "\n"])
    else:
        parts.extend([captions, "\n"])
    
    return "".join(parts)



def build_context(item, include_reasoning=False, up_to=None):
    """
    Construct progressive reference context for the user prompt.
    Args:
        item (dict): Input data item
        include_reasoning (bool): Whether to include reasoning in context
        up_to (str): Control context amount: None, 'caption', or 'text2img'
    Returns:
        str: Constructed context text
    """
    # Use list for more efficient string concatenation
    parts = ["\nReference Context:\n"]
    parts.extend([item["summary"], "\n"])
    parts.append(format_captions(item["caption"]))

    # Simplified condition logic
    if up_to is None or up_to == "text2img":
        parts.extend([item["text2img"], "\n"])
    if up_to in (None, "caption", "text2img"):
        parts.extend([item["img2img"], "\n"])

    if include_reasoning and "reasoning" in item:
        parts.extend([item["reasoning"], "\n"])
    
    return "".join(parts)



# Example usage
if __name__ == "__main__":
    try:
        # Use local test data file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, "test_data.json")
        
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Test all epochs to ensure functionality
        for epoch in range(5):
            print(f"\n=== Testing epoch {epoch} ===")
            example = normal_data_transform(data[0], epoch=epoch)
            print(json.dumps(example, indent=2, ensure_ascii=False))
        
    except FileNotFoundError:
        print("Error: Data file not found. Please check the path.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON file.")
    except Exception as e:
        print(f"Error: {str(e)}")
