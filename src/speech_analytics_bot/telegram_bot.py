from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters
import filetype
import logging

token_file = open('TOKEN.txt')
BOT_TOKEN = token_file.read()

updater = Updater(token=BOT_TOKEN, use_context=True)

dispatcher = updater.dispatcher

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)

def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Hi, I`m CallCenterAnalyticsBot. Ready to get dialog recording(in .wav) and return report about operator work. ")

def wav_downloader(update, context):
    context.bot.get_file(update.message.document).download()
    context.bot.send_message(chat_id=update.effective_chat.id, text="File for analysis received")
    with open("../../data/file.wav", 'wb') as f:
        context.bot.get_file(update.message.document).download(out=f)
    context.bot.send_message(chat_id=update.effective_chat.id, text="File saved")
    context.bot.send_message(chat_id=update.effective_chat.id, text="Starting processing")

#def convert_wav():

start_handler = CommandHandler('start', start)

def echo(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)

echo_handler = MessageHandler(Filters.text & (~Filters.command), echo)

dispatcher.add_handler(echo_handler)
dispatcher.add_handler(start_handler)
dispatcher.add_handler(MessageHandler(Filters.document.category('audio/'), wav_downloader))

updater.start_polling()
# updater.idle()