# Muchanizer

Muchanizer is an AI image transformation system that turns regular photos into Alphonse Mucha-inspired artistic images. It consists of a Telegram bot interface that allows users to easily transform their images with a simple message. It also includes manifests to run this on OpenShift.

## Overview

The project uses Stable Diffusion XL with ControlNet for image-to-image generation, running on a KServe infrastructure. The system extracts edge information from submitted images and generates new images in the style of Alphonse Mucha, characterized by flowing lines, intricate patterns, and decorative motifs.

## Architecture

Muchanizer consists of two main components:

1. **Telegram Bot** - Handles user interactions and image processing requests
2. **Model Server** - Runs the AI model that performs the actual image transformation

```
┌─────────────┐         ┌─────────────────┐
│             │ Request │                 │
│ Telegram    │────────>│ Model Server    │
│ Bot         │         │ (SDXL+ControlNet)│
│             │<────────│                 │
└─────────────┘ Response└─────────────────┘
```

## Components

### Telegram Bot

The bot is built using the `python-telegram-bot` library and accepts image uploads from users. When a user sends an image, the bot:

1. Downloads the image
2. Sends it to the Model Server for processing
3. Returns the transformed image to the user

Users can optionally include a custom prompt as the image caption, or the system will use a default Alphonse Mucha style prompt.

### Model Server

The model server uses:

- Stable Diffusion XL as the base model
- ControlNet with Canny edge detection for conditioning
- KServe for model serving

The server detects edges in the uploaded image, then generates a new image that follows those edges but in the Alphonse Mucha art style.

## Setup & Installation

### Prerequisites

- Docker
- Python 3.12+
- A Telegram bot token (created via BotFather)
- Access to GPU resources (recommended)

### Environment Variables

#### Bot

Create a `.env` file in the bot directory with:

```
TELEGRAM_TOKEN=your_telegram_bot_token
MODEL_ENDPOINT=http://modelserver:8080/v2/models/infer/infer
MODEL_TOKEN=optional_auth_token
GUIDANCE_SCALE=3.0
STRENGTH=0.25
```

#### Model Server

The model server needs these environment variables:

```
MODEL_ID=/mnt/models
CONTROLNET_MODEL=lllyasviel/sd-controlnet-canny
LORA_MODEL=optional_lora_path
LORA_WEIGHT_NAME=optional_lora_weight
```

### Docker Installation

1. Build the bot image:
   ```
   cd muchanizer/src/bot
   docker build -t muchanizer-bot .
   ```

2. Build the model server image:
   ```
   cd muchanizer/src/modelserver
   docker build -t muchanizer-modelserver .
   ```

3. Run the containers:
   ```
   docker run -d --name modelserver --gpus all -v /path/to/models:/mnt/models muchanizer-modelserver
   docker run -d --name bot --link modelserver:modelserver -e MODEL_ENDPOINT=http://modelserver:8080/v2/models/infer/infer muchanizer-bot
   ```

## Usage

1. Start a chat with your Telegram bot
2. Send an image to the bot
3. Optionally include a custom prompt as the image caption
4. Receive the transformed image in Alphonse Mucha style

### Default Style

If no custom prompt is provided, the system applies the default Alphonse Mucha style:

```
Alphonse Mucha Style, characterized by flowing lines, intricate patterns, and decorative motifs.
Include elements like delicate floral patterns, ethereal hair adorned with flowers or jewels,
and ornate accessories to enhance the royal aesthetic. Intense Mucha style background.
Thick Lines, high-contrast, high-resolution high definition in details and ink outlines,
no shading, intricate detailed frame border
```

## Customization

### Adjusting Parameters

You can customize the generation parameters by modifying:

- `GUIDANCE_SCALE`: Controls how closely the output follows the prompt (default: 3.0)
- `STRENGTH`: Controls how much the original image influences the result (default: 0.25)

### Custom Prompts

Users can include custom prompts as captions when sending images to the bot. These prompts will override the default style.

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](LICENSE) file for details or visit [https://www.gnu.org/licenses/gpl-3.0.en.html](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Acknowledgements

This project utilizes:
- Hugging Face's Diffusers library
- Python Telegram Bot
- KServe for model serving
