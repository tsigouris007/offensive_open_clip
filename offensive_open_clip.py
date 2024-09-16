#!/usr/bin/env python
import open_clip
from PIL import Image
import torch
import json
import argparse
from pathlib import Path
import sys
import requests
import time
from tabulate import tabulate
from urllib.parse import urlparse, unquote
from io import BytesIO
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

ALLOWED_IMG_FORMAT = ('.jpg', '.jpeg', '.png')

VIOLENCE_TEXT_LABELS = [
    "violence", "blood", "gore", "death", "murder", "assault", "fighting", "abuse",
    "gun", "weapon", "knife", "explosion", "war", "terrorism", "terrorist", "bomb",
    "mass shooting", "riot", "kidnapping", "hostage", "shooting", "execution",
    "war tank", "nuke",
]

SEXUAL_TEXT_LABELS = [
    "nudity", "sexual content", "explicit content", "profanity", "pornographic content", "porn",
    "sex", "rape", "molestation", "incest", "sexual harassment", "fetish", "child pornography",
    "ass", "asshole", "vagina", "pussy", "genitals", "dick", "boobs", "dildo", "gag", "sex toy", "anal",
    "fuck", "fucking", "pussy fucking", "ass fucking", "semen", "poop",
]

DRUGS_TEXT_LABELS = [
    "drugs", "drug use", "alcohol", "smoking", "addiction",
    "cocaine", "heroin", "marijuana", "opioids", "meth", "drug paraphernalia",
    "overdose", "pill", "syringe", "injection",
]

RACISM_TEXT_LABELS = [
    "racism", "hate speech", "offensive language", "discrimination",
    "sexism", "homophobia", "transphobia", "slur", "xenophobia", "anti-semitism",
    "misogyny", "bigotry", "white supremacy", "neo-nazi", "alt-right", "extremism",
    "hate crime", "terror group", "cult", "propaganda",
]

HARASSMENT_TEXT_LABELS = [
    "harassment", "bullying", "cyberbullying", "stalking",
    "child abuse", "animal abuse", "self-harm", "suicide", "sadism", "torture",
    "domestic violence", "spousal abuse", "verbal abuse", "psychological abuse",
]

ILLEGAL_ACTIVITIES_TEXT_LABELS = [
    "theft", "vandalism", "fraud", "money laundering", "bribery",
    "extortion", "blackmail", "corruption", "scam", "hacking", "phishing",
    "identity theft", "counterfeiting", "illegal trade", "human trafficking",
    "organized crime", "drug trafficking", "arms trafficking", "terrorism financing",
    "arson", "looting", "tax evasion", "embezzlement", "racketeering", "prohibited goods",
]

PII_TEXT_LABELS = [
    "passport", "ID card", "driver's license", "social security number",
    "credit card", "bank statement", "birth certificate", "visa", "medical record",
    "address", "phone number", "email address", "biometric data", "fingerprint",
    "face recognition", "personal document", "tax document", "insurance card",
]

OTHER_TEXT_LABELS = [
    "disease", "epidemic", "plague", "biological weapon", "biohazard",
    "vomit", "feces", "urine", "defecation", "rotting",
    "decay", "severed", "decomposition", "corpse", "dead body", "autopsy",
    "abnormal", "demonic", "occult", "ritual", "witchcraft", "satanic",
    "cult", "blood", "sacrifice", "sacrilege",
]

DISALLOWED_TEXT_LABELS = VIOLENCE_TEXT_LABELS + SEXUAL_TEXT_LABELS + DRUGS_TEXT_LABELS + RACISM_TEXT_LABELS + HARASSMENT_TEXT_LABELS + ILLEGAL_ACTIVITIES_TEXT_LABELS + PII_TEXT_LABELS + OTHER_TEXT_LABELS

def list_pretrained_models():
    models = open_clip.list_pretrained()
    
    # Prepare data for the table
    table_data = []
    for model_name, pretrained_model in models:
        table_data.append([model_name, pretrained_model])

    # Print the table
    print(tabulate(table_data, headers=["Model", "Pretrained Version"], tablefmt="grid"))


def runs_on():
    # Run on GPU if available else on CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def predict_label(device, model, tokenizer, image, image_name):
    # Disallowed labels passed here
    text = tokenizer(DISALLOWED_TEXT_LABELS)

    start_time = time.time()

    # Calculate stuff
    with torch.no_grad(), torch.amp.autocast(device):
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    end_time = time.time()

    # Find the max index label
    max_prob_index = text_probs.argmax(dim=-1).item()

    # Get the label
    predicted_label = DISALLOWED_TEXT_LABELS[max_prob_index]

    # Humanize the probability
    predicted_probability = text_probs[0, max_prob_index].item()

    result = {
        "image": str(image_name),
        "label": predicted_label,
        "probability": round(predicted_probability, 4),
        "prediction_time": round(end_time - start_time, 4)
    }

    return json.dumps(result, indent=4)


def evaluate_label(label, threshold):
    # Parse the JSON label result
    label_dict = json.loads(label)
    
    # Check if the probability exceeds the threshold
    if label_dict["probability"] > threshold:
        label_dict["state"] = 'disallowed'
    else:
        label_dict["state"] = 'allowed'
    
    # Return the updated label with the new state field as a JSON string
    return json.dumps(label_dict, indent=4)



def main(selected_model, selected_dataset, image_path, image_url, image_dir, threshold, device):
    # Create the model and transformations
    model, _, preprocess = open_clip.create_model_and_transforms(selected_model, pretrained=selected_dataset)

    # This disables model training mode
    model.eval()

    # Use the corresponding tokenizer
    tokenizer = open_clip.get_tokenizer(selected_model)

    # Read the image
    if image_path:
        image = preprocess(Image.open(image_path)).unsqueeze(0)
        # Predict
        label = predict_label(device, model, tokenizer, image, image_path)
        # Evaluate
        print(evaluate_label(label, threshold))
    elif image_url:
        # Fetch from URL
        response = requests.get(image_url)
        if response.status_code == 200:
            image = preprocess(Image.open(BytesIO(response.content))).unsqueeze(0)
            # Predict
            label = predict_label(device, model, tokenizer, image, image_url)
            # Evaluate
            print(evaluate_label(label, threshold))
        else:
            print(f"Failed to download image from '{image_url}'.")
            sys.exit(1)
    elif image_dir:
        image_files = [file for file in Path(image_dir).glob('*') if file.suffix.lower() in ALLOWED_IMG_FORMAT]
        if not image_files:
            print(f"Error: No files with {ALLOWED_IMG_FORMAT} found in {image_dir}.")
            sys.exit(1)
        for image_file in image_files:
            image = preprocess(Image.open(image_file)).unsqueeze(0)
            # Predict
            label = predict_label(device, model, tokenizer, image, image_file)
            # Evaluate
            print(evaluate_label(label, threshold), flush=True)


def valid_threshold(value):
    # Convert the input value to float and check if it's between 0.0 and 1.0
    try:
        fvalue = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid threshold value: {value}. Must be a float.")
    
    if fvalue < 0.0 or fvalue > 1.0:
        raise argparse.ArgumentTypeError(f"Threshold must be between 0.0 and 1.0. Got {value}.")
    
    return fvalue


def is_valid_image_url(url):
    parsed_url = urlparse(url)

    # Check if the URL starts with http or https
    if parsed_url.scheme not in ('http', 'https'):
        return False

    path = unquote(parsed_url.path)
    return path.lower().endswith((ALLOWED_IMG_FORMAT))


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    # Mutually exclusive group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image-path", type=str, help="Local path to the image file to be processed.")
    group.add_argument("--image-url", type=str, help="URL of the image to be downloaded and processed.")
    group.add_argument("--image-dir", type=str, help="Directory containing images to be processed.")
    group.add_argument("--list-models", action="store_true", help="List all available pretrained models.")
    group.add_argument("--runs-on", action="store_true", help="Show if running on cuda-GPU or CPU. GPU implementations are running faster.")
    parser.add_argument("--threshold", type=valid_threshold, default=0.45, help="Probability threshold for disallowing content (must be between 0.0 and 1.0).")
    parser.add_argument("--model", type=str, default="ViT-B-32", help="The model to use for predictions. Use '--list-models' for available models. Defaults to 'ViT-B-32'.")
    parser.add_argument("--dataset", type=str, default="laion2b_s34b_b79k", help="The pretrained dataset to use for predictions. Use '--list-models' for available datasets. Defaults to 'laion2b_s34b_b79k'.")

    args = parser.parse_args()

    if args.list_models:
        list_pretrained_models()
        sys.exit(0)

    if args.runs_on:
        device = runs_on()
        print(f"Running on '{device}'.")
        sys.exit(0)

    if args.image_path:
        if not Path(args.image_path).exists():
            print(f"No such image file '{args.image_path}'")
            sys.exit(1)
        if not args.image_path.lower().endswith(ALLOWED_IMG_FORMAT):
            print(f"Error: Only {ALLOWED_IMG_FORMAT} files allowed.")
            sys.exit(1)

    if args.image_url:
        if not is_valid_image_url(args.image_url):
            print(f"Error: Only {ALLOWED_IMG_FORMAT} files allowed.")
            sys.exit(1)

    if args.image_dir:
        if not Path(args.image_dir).is_dir():
            print(f"No such directory '{args.image_dir}'.")
            sys.exit(1)

    main(args.model, args.dataset, args.image_path, args.image_url, args.image_dir, args.threshold, runs_on())
