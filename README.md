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
![Example answer](Example%20answer.png)