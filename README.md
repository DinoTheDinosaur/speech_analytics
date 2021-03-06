# speech_analytics
Speech analytics package for call-center allows you to estimate your operator's work. It provides a user-friendly interface using Telegram bots.

Every file goes through the following operations:
* Noise suppression
* Diarization
* Voice activity detection
* Pauses and interruptions detection
* Necessary phrases detection
* Bad words and threats detection
![System Architecture](https://github.com/DinoTheDinosaur/speech_analytics/blob/main/System_architecture-Page-1.png?raw=true)

# Installing
First, you need to clone the repository:
```
git clone https://github.com/DinoTheDinosaur/speech_analytics
cd speech_analytics
```

Create virtual environment and install necessary packages. Please note, that **only Python 3.8** is supported:
```
python -m venv venv
python -m pip install -r requirements.txt
```

Create config.yaml file:
```yaml
# for SR (yandex or vosk)
recognition_engine: "vosk"

# necessary files (models, corpuses, etc.)
suppressor_model_weights: "./data/model_weights.ckpt"
vosk_model: "./data/vosk"
white_list: "./data/white_list.json"
obscene_corpus: "./data/obscene_corpus.json"
threats_corpus: "./data/threats_corpus.json"
white_checklist: "./data/check_list_white.json"
black_checklist: "./data/check_list_black.json"

# for Telegram bot
bot_token: "YOUR_TELEGRAM_BOT_ACCESS_TOKEN_HERE"

# for Yandex Speech kit
bucket: "YOUR_BUCKET_HERE"
aws_key: "YOUR_AWS_KEY_HERE"
aws_key_id: "YOUR_AWS_KEY_ID_HERE"
ya_api_key: "YOUR_YANDEX_SPEECH_KIT_API_KEY"
```

Run bot:
```
python run_bot.py
```

# Example answer
![Example answer](https://github.com/DinoTheDinosaur/speech_analytics/blob/develop/Example%20answer.png)

# Models
In case you didn't find necessary data in the repository, you can download it directly:
* [Noise suppressor model weights](https://drive.google.com/file/d/1Ih8pZ3n4i6VXgwKFYQfMWu3PwCiPtgpG/view)
* [Vosk model](https://alphacephei.com/vosk/models)

# CompTech 2021
* [Технический проект](https://docs.google.com/document/d/1jLiDmdPZaRlNvCNtiNmEHCS2ZOK0XDdi7ywtUssIlNM/edit?usp=sharing)
* [Техническое задание](https://docs.google.com/document/d/1zfrNlRrlqfpwu3aPvmqAVw_UwfJXdKb8wJ2hC7-WKU4/edit?usp=sharing)
* [Руководство пользователя](https://docs.google.com/document/d/1xItH4Xq1IK36KrAj7ykrxKjiqRLaTJUcrGsqtBjqwRk/edit?usp=sharing)