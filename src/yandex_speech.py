import boto3
import requests
import time


def yandex_speech(input_link: str, bucket: str, aws_secret_access_key: str,
                  aws_access_key_id: str, API_key: str) -> str:
    """
    :param input_link: path to the wav-file
    :param bucket: name of bucket at Yandex.Cloud
    :param aws_secret_access_key: secret code of the access key
    :param aws_access_key_id: ID of the access key
    :param API_key: ID of the API_key
    :return:
    """
    # загрузка в облако
    session = boto3.session.Session(region_name="ru-central1", aws_secret_access_key=aws_secret_access_key,
                                    aws_access_key_id=aws_access_key_id)
    s3 = session.client(service_name='s3', endpoint_url='https://storage.yandexcloud.net/')
    s3.upload_file(input_link, bucket, 'obj.wav')\

    file_link = 'https://storage.yandexcloud.net/' + bucket + '/obj.wav'
    POST = "https://transcribe.api.cloud.yandex.net/speech/stt/v2/longRunningRecognize"

    body = {
        "config": {
            "specification": {
                "languageCode": "ru-RU",
                "model": "general:rc",
                "audioEncoding": "LINEAR16_PCM",
                "sampleRateHertz": 8000,
                "audioChannelCount": 1
            }
        },
        "audio": {
            "uri": file_link
        }
    }

    header = {'Authorization': 'Api-Key {}'.format(API_key)}

    # Отправить запрос на распознавание.
    req = requests.post(POST, headers=header, json=body)
    data = req.json()
    id_ = data['id']

    GET = "https://operation.api.cloud.yandex.net/operations/" + str(id_)

    # Запрашивать на сервере статус операции, пока распознавание не будет завершено.
    wait = 0
    while wait <= 10:

        time.sleep(1)
        wait += 1

        req = requests.get(GET, headers=header).json()

        if req['done']:
            break

    # Текст результатов распознавания.
    output_text = ''
    for chunk in req['response']['chunks']:
        c = chunk['alternatives'][0]['text']
        output_text += ' ' + c

    # удаление объекта в облаке
    for_deletion = [{'Key': 'obj.wav'}]
    s3.delete_objects(Bucket=bucket, Delete={'Objects': for_deletion})

    return output_text
