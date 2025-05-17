import os
import sys
from dotenv import load_dotenv, find_dotenv
from PIL import Image

from typing import Final

from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

import structlog

from ai import img2img_pipeline

log = structlog.get_logger()
log.info('Starting up bot...')

load_dotenv(find_dotenv())

TOKEN: Final = os.getenv('TELEGRAM_TOKEN')
if not TOKEN:
    log.info("Bot token is not set")
    sys.exit(1)

DEFAULT_PROMPT = """Alphonse Mucha Style, characterized by flowing lines, intricate patterns, and decorative motifs. Include elements like delicate floral patterns, ethereal hair adorned with flowers or jewels, and ornate accessories to enhance the royal aesthetic. Intense Mucha style background. Thick Lines, high-contrast, high-resolution high definition in details and ink outlines, no shading, intricate detailed frame border"""

async def start_command(update: Update, context:ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    await update.message.reply_text("Hello!")

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')

async def downloader(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Download file
    new_file = await update.message.effective_attachment[-1].get_file()
    file = await new_file.download_to_drive()

    return file

async def delete_file_from_drive(file):
    os.remove(file)

async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if (
            not update.message
            or not update.effective_chat
            or (
                not update.message.photo
                and not update.message.video
                and not update.message.document
                and not update.message.sticker
                and not update.message.animation
            )
        ):
            print(update)
            return

    log.info('Fetching image', id=update.update_id)
    file = await downloader(update, context)

    if not file:
        await update.message.reply_text("Something went wrong, try again")
        return

    log.info('Generating new image', id=update.update_id)
    image = Image.open(file)
    if update.message.caption:
        prompt = update.message.caption
    else:
        prompt = DEFAULT_PROMPT

    new_image = await img2img_pipeline(image, prompt)

    await delete_file_from_drive(file)
    await update.message.reply_photo(new_image)


def scale_and_paste(original_image):
    aspect_ratio = original_image.width / original_image.height

    if original_image.width > original_image.height:
        new_width = 1024
        new_height = round(new_width / aspect_ratio)
    else:
        new_height = 1024
        new_width = round(new_height * aspect_ratio)

    resized_original = original_image.resize((new_width, new_height), Image.LANCZOS)
    return resized_original

if __name__ == '__main__':
    app = Application.builder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, generate_image))
    # app.add_error_handler(error)
    app.run_polling(poll_interval=3)
