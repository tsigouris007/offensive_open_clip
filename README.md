# offensive_open_clip
A ready to use open clip implementation that aims to catch offensive image file uploads

# Why

This repository was inspired to be used as an image upload filtering mechanism by utilizing https://github.com/mlfoundations/open_clip to avoid inconvenient uploads by malicious users. This will also ease out the pain of reviewing all images (if this is something that you do in your daily lives or business).

# How

It is ready to use with the capability to scan a local image path, an image from a URL or a local directory.

First install via requirements or follow the `instructions.txt` file (prefer this way):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install requests tabulate
pip install open-clip-torch==2.26.1
```

The run the help dialogue:

```bash
$ ./offensive_open_clip.py -h
usage: offensive_open_clip.py [-h]
                              (--image-path IMAGE_PATH | --image-url IMAGE_URL | --image-dir IMAGE_DIR | --list-models | --runs-on)
                              [--threshold THRESHOLD] [--model MODEL]
                              [--dataset DATASET]

options:
  -h, --help            show this help message and exit
  --image-path IMAGE_PATH
                        Local path to the image file to be processed.
  --image-url IMAGE_URL
                        URL of the image to be downloaded and processed.
  --image-dir IMAGE_DIR
                        Directory containing images to be processed.
  --list-models         List all available pretrained models.
  --runs-on             Show if running on cuda-GPU or CPU. GPU
                        implementations are running faster.
  --threshold THRESHOLD
                        Probability threshold for disallowing content (must be
                        between 0.0 and 1.0).
  --model MODEL         The model to use for predictions. Use '--list-models'
                        for available models. Defaults to 'ViT-B-32'.
  --dataset DATASET     The pretrained dataset to use for predictions. Use '--
                        list-models' for available datasets. Defaults to
                        'laion2b_s34b_b79k'.
```

# Models and datasets

Use the following command to list:

```bash
$ ./offensive_open_clip.py --list-models
+----------------------------+------------------------------------+
| Model                      | Pretrained Version                 |
+============================+====================================+
| RN50                       | openai                             |
+----------------------------+------------------------------------+
| RN50                       | yfcc15m                            |
+----------------------------+------------------------------------+
| RN50                       | cc12m                              |
+----------------------------+------------------------------------+
| RN50-quickgelu             | openai                             |
+----------------------------+------------------------------------+
| RN50-quickgelu             | yfcc15m                            |
+----------------------------+------------------------------------+
| RN50-quickgelu             | cc12m                              |
+----------------------------+------------------------------------+
| RN101                      | openai                             |
+----------------------------+------------------------------------+
| RN101                      | yfcc15m                            |
+----------------------------+------------------------------------+
| RN101-quickgelu            | openai                             |

...

+----------------------------+------------------------------------+
| ViTamin-L2-256             | datacomp1b                         |
+----------------------------+------------------------------------+
| ViTamin-L2-336             | datacomp1b                         |
+----------------------------+------------------------------------+
| ViTamin-L2-384             | datacomp1b                         |
+----------------------------+------------------------------------+
| ViTamin-XL-256             | datacomp1b                         |
+----------------------------+------------------------------------+
| ViTamin-XL-336             | datacomp1b                         |
+----------------------------+------------------------------------+
| ViTamin-XL-384             | datacomp1b                         |
+----------------------------+------------------------------------+
```

The default one does its job pretty well. You can also specify another model, dataset and threshold as shown:

```bash
$ ./offensive_open_clip.py --image-path=img/bank1.png --model=ViT-bigG-14 --dataset=laion2b_s39b_b160k --threshold=0.4
```

# Output

The output of an image scan will have the following format:

```bash
$ ./offensive_open_clip.py --image-path=img/ass1.jpg
{
    "image": "img/ass1.jpg",
    "label": "ass",
    "probability": 0.9341,
    "prediction_time": 3.837,
    "state": "disallowed"
}

$ ./offensive_open_clip.py --image-path=img/product1.jpeg
{
    "image": "img/product1.jpeg",
    "label": "bomb",
    "probability": 0.2271,
    "prediction_time": 3.809,
    "state": "allowed"
}
```

If the state is `disallowed` then it means we are on to something. You can verify by running some samples in the `img/` directory.

# Notes

Larger models and datasets come with better results so you have to balance out speed with accuracy and tune out your prefered threshold. \
I do not own the models or datasets. This is simply a wrapper that utilizes the already existing `open_clip` tool with a few tweaks and enhancements.
