#!/usr/bin/env python
# coding: utf-8



# -*- coding: utf-8 -*-

import requests
import time
import json

# Укажите ваш API-ключ и ссылку на аудиофайл в Object Storage.
key = ''
filelink = ''

POST = "https://transcribe.api.cloud.yandex.net/speech/stt/v2/longRunningRecognize"

body ={
    "config": {
        "specification": {
            "languageCode": "ru-RU",
            "model": "general:rc",
            "audioEncoding": "LINEAR16_PCM",
            "sampleRateHertz": 48000,
            "audioChannelCount": 1
        }
    },
    "audio": {
        "uri": filelink
    }
}

header = {'Authorization': 'Api-Key {}'.format(key)}

# Отправить запрос на распознавание.
req = requests.post(POST, headers=header, json=body)
data = req.json()
print(data)

id = data['id']

# Запрашивать на сервере статус операции, пока распознавание не будет завершено.
while True:

    time.sleep(1)

    GET = "https://operation.api.cloud.yandex.net/operations/{id}"
    req = requests.get(GET.format(id=id), headers=header)
    req = req.json()

    if req['done']: break
    print("Not ready")


# Текст результатов распознавания.
print("Text chunks:")
for chunk in req['response']['chunks']:
    print(chunk['alternatives'][0]['text'])







