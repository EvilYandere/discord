import discord
import os
import shutil
from discord.ext import commands
from discord.ext.commands import has_permissions, MissingPermissions
import uuid
import requests

import cv2
import numpy as np

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.oauth2 import service_account
from Google import Create_Service
from fuzzywuzzy import fuzz
from nltk.stem.snowball import RussianStemmer
import collections
import re


import easyocr


from deep_translator import GoogleTranslator


TOKEN = "some_ds_token"

client = commands.Bot(command_prefix="&", intents=discord.Intents.all())

#   print("biba")

@client.event
async def on_ready():
    print("Бот готов")


@client.event
async def on_member_join(member):
    await member.create_dm()
    await member.dm_channel.send(
        f"Привет, {member.name}! \n"
        "Если не знаешь, что делать, набери &about \n"
        "Со мной можно общаться как здесь, так и на общем канале"
    )


@client.command(aliases=['purge', 'delete'])
@commands.has_permissions(manage_messages=True)
async def clear(ctx, amount=None):
    if amount is None:
        print("Очищаю чат...")
        await ctx.channel.purge(limit=1000000)


@clear.error
async def clear_error(ctx, error):
    if isinstance(error, MissingPermissions):
        await ctx.send("По всей видимости, у тебя нет прав  для этой команды")


@client.command()
async def about(ctx):
    await ctx.send("Если что, все команды вызываются через символ & и пишутся буквами в нижнем регистре \n"
                   "Вот, что я умею: \n"
                   "detect_obj - напиши это в качестве сообщения к картинке, которую нужно анализировать "
                   "на предмет объектов на ней \n"
                   "detect_text - распознать текст на фотографии. Правила те же, что и для "
                   "обнаружения объектов \n"
                   "video - скинуть обучающее видео \n"
                   "Для дополнительной информации о команде набери video_help \n"
                   "translate - перевести сообщение \n"
                   "Для дополнительной информации о команде набери translate_help")


@client.command()
async def video_help(ctx):
    await ctx.send("Для начала вот список языков: c#, c++, java, python \n"
                   "Чтобы получить видео по теме, набери сначала язык, а потом в двух-трех словах нужную тему \n"
                   "Например: &video python keras \n"
                   "Примечание - для улучшения результата постарайся использовать ключевые слова")


@client.command()
async def translate_help(ctx):
    await ctx.send("Чтобы перевести фразу, набери сначала язык, на который хочешь перевести текст,"
                   " а потом - сам текст \n"
                   "Например: &translate russian hello world!")


@client.command()
async def detect_obj(ctx):
    try:
        url = ctx.message.attachments[0].url
    except IndexError:
        print("Картинка не найдена")
        await ctx.send("Картинка не обнаружена")
    else:
        if url[0:26] != "https://cdn.discordapp.com":
            await ctx.send("Картинка обнаружена, но discord неправильно сгенерировал ссылку. \n"
                           "Попробуй еще раз, или выбери другое изображение")
            return
        else:
            if url.endswith('png') or url.endswith('jpg') or url.endswith('jpeg'):

                r = requests.get(url, stream=True)
                imageName = str(uuid.uuid4()) + ".jpg"
                with open(imageName, 'wb') as out_file:
                    shutil.copyfileobj(r.raw, out_file)
                    print(f"Картинка {imageName} сохранена")
                    await ctx.send("Картинка сохранена, начинаю обработку")

                    result = []

                def get_output_layers(net):
                    layer_names = net.getLayerNames()
                    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
                    return output_layers

                def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
                    label = str(classes[class_id])
                    color = COLORS[class_id]
                    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
                    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    result.append((label, confidence))

                image = cv2.imread(imageName)
                try:
                    Width = image.shape[1]
                    Height = image.shape[0]
                except AttributeError:
                    print("Не удается открыть изображение")
                    await ctx.send("Не удается открыть изображение")
                    os.unlink(imageName)
                    return



                scale = 0.00392

                classes = None

                with open("yolov3.txt", "r") as f:
                    classes = [line.strip() for line in f.readlines()]

                COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
                blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(get_output_layers(net))
                class_ids = []
                confidences = []
                boxes = []
                conf_threshold = 0.5
                nms_threshold = 0.4

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            center_x = int(detection[0] * Width)
                            center_y = int(detection[1] * Height)
                            w = int(detection[2] * Width)
                            h = int(detection[3] * Height)
                            x = center_x - w / 2
                            y = center_y - h / 2
                            class_ids.append(class_id)
                            confidences.append(float(confidence))
                            boxes.append([x, y, w, h])

                try:
                    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
                except Exception as ex:
                    print(ex)
                    await ctx.send("Что-то пошло не так при анализе")
                    try:
                        os.unlink(imageName)
                        os.unlink(f"{imageName}_res.jpg")
                    except Exception as ex:
                        print(ex)
                    return


                for i in indices:
                    box = boxes[i]
                    x = box[0]
                    y = box[1]
                    w = box[2]
                    h = box[3]
                    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

                if len(result) == 0:
                    print("Для этой картинки нет результатов")
                    await ctx.send("К сожалению, для этой картинки результатов нет")
                else:
                    cv2.imwrite(f"{imageName}_res.jpg", image)
                    await ctx.send("Вот, что удалось найти:")
                    await ctx.send(file=discord.File(f"{imageName}_res.jpg"))
                    await ctx.send("А вот список объектов и точность обнаружения:")
                    for k in result:
                        await ctx.send(f"{k}")
                    await ctx.send("Обработка окончена!")
                try:
                    os.unlink(imageName)
                    os.unlink(f"{imageName}_res.jpg")
                except Exception as ex:
                    print(ex)

            else:
                print("Это не картинка")
                await ctx.send("Это не картинка")
                return


@client.command()
async def video(ctx):
    msg_content = str(ctx.message.content)
    lang_list = ["c#", "c++", "java", "python"]
    if len(msg_content.rstrip()[6:]) == 0:
        await ctx.send("Укажи необходимые параметры. При необходимости обратись к video_help")
        return
    lang = msg_content.rstrip()[6:].split()[0]
    if lang not in lang_list:
        await ctx.send("Выбери язык, что есть в списке")
        return
    qstn = " ".join(msg_content.rstrip()[6:].split()[1:])
    if qstn == "":
        await ctx.send("Напиши текст запроса")
        return

    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("mycreds.txt")
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile("mycreds.txt")
    drive = GoogleDrive(gauth)

    CLIENT_SECRET_FILE = "client_secrets.json"
    API_NAME = "drive"
    API_VERSION = "v3"
    SCOPES = ["https://www.googleapis.com/auth/drive"]

    service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

    if lang == "c#":
        lang_id = "1dTvAdonnFK0szyE_8sLHuxq3rcHuljZ2"
    if lang == "c++":
        lang_id = "1VWNcLsyfE3eH8g609JhLv7Tf3ZRHS2aL"
    if lang == "java":
        lang_id = "1WelfgA7qU_S40MXkmBeHXeCrrXkVazI_"
    if lang == "python":
        lang_id = "1ROiYkrqdpxJXfJQsH_OgI2ECyXEEWhxs"

    fileList = drive.ListFile({'q': f"'{lang_id}' in parents and trashed=false"}).GetList()

    stemmer = RussianStemmer()

    ask = re.sub(r'[^\w\s]','', qstn)
    ask_list = ask.split()
    snow_ask_list = [stemmer.stem(x) for x in ask_list]
    ask_str = " ".join(snow_ask_list)

    if len(fileList) == 0:
        print("На диске нет файлов")
        await ctx.send("На диске почему-то нет файлов")
        exit(1)
    else:
        pairs_arr = []
        result_dict = {}
        for dict in fileList:
            id = dict.get("id")
            title = dict.get("title")[:-4]
            pairs_arr.append((id, re.sub(r'[^\w\s]','', title)))
        for i in range(len(pairs_arr)):
            snow_pairs_arr_list = [stemmer.stem(x) for x in pairs_arr[i][1].split()]
            pairs_str = " ".join(snow_pairs_arr_list)
            percent = fuzz.token_set_ratio(ask_str, pairs_str)
            if percent >= 40:
                result_dict[f"{i}"] = percent

        if len(result_dict) == 0:
            await ctx.send("Подходящих результатов нет")
            return

        sorted_dict = {}
        sorted_keys = sorted(result_dict, key=result_dict.get)

        for w in sorted_keys:
            sorted_dict[w] = result_dict[w]

        [last] = collections.deque(sorted_dict, maxlen=1)
        result_url = ''
        for i, item in enumerate(pairs_arr):
            if i == int(last):
                result_url = item[0]

        file_id = f"{result_url}"

        request_body = {
            "role": "reader",
            "type": "anyone"
        }

        response_permission = service.permissions().create(
            fileId=file_id,
            body=request_body
        ).execute()

        response_share_link = service.files().get(
            fileId=file_id,
            fields='webViewLink'
        ).execute()

        answer = response_share_link.get("webViewLink")

        await ctx.send("Вот видео, которое больше всего подходит под твой запрос:")
        await ctx.send(f"{answer}")

        copy_dict = result_dict
        del copy_dict[last]
        if len(copy_dict) > 0:
            new_sorted_dict = {}
            new_sorted_keys = sorted(copy_dict, key=copy_dict.get)

            for w in new_sorted_keys:
                new_sorted_dict[w] = copy_dict[w]

            [new_last] = collections.deque(new_sorted_dict, maxlen=1)
            new_result_url = ''
            for i, item in enumerate(pairs_arr):
                if i == int(new_last):
                    new_result_url = item[0]

            file_id = f"{new_result_url}"

            request_body = {
                "role": "reader",
                "type": "anyone"
            }

            response_permission = service.permissions().create(
                fileId=file_id,
                body=request_body
            ).execute()

            response_share_link = service.files().get(
                fileId=file_id,
                fields='webViewLink'
            ).execute()

            new_answer = response_share_link.get("webViewLink")

            await ctx.send("А вот еще одно видео, которое так жеможет оказаться полезным:")
            await ctx.send(f"{new_answer}")


@client.command()
async def detect_text(ctx):
    try:
        url = ctx.message.attachments[0].url
    except IndexError:
        print("Картинка не найдена")
        await ctx.send("Картинка не обнаружена")
    else:
        if url[0:26] != "https://cdn.discordapp.com":
            await ctx.send("Картинка обнаружена, но discord неправильно сгенерировал ссылку. \n"
                           "Попробуй еще раз, или выбери другое изображение")
            return
        else:
            if url.endswith('png') or url.endswith('jpg') or url.endswith('jpeg'):
                r = requests.get(url, stream=True)
                imageName = str(uuid.uuid4()) + ".jpg"
                with open(imageName, 'wb') as out_file:
                    shutil.copyfileobj(r.raw, out_file)
                    print(f"Картинка {imageName} сохранена")
                    await ctx.send("Картинка сохранена, начинаю обработку")
                img = cv2.imread(imageName)
                try:
                    Width = img.shape[1]
                    Height = img.shape[0]
                except AttributeError:
                    print("Не удается открыть изображение")
                    await ctx.send("Не удается открыть изображение")
                    os.unlink(imageName)
                    return
                reader = easyocr.Reader(['ru', 'en'], gpu=True)
                try:

                    result_text = reader.readtext(img, detail=0, paragraph=True)
                except Exception as ex:
                    print(ex)
                    await ctx.send("Что-то пошло не так при анализе")
                    try:
                        os.unlink(imageName)
                    except Exception as ex:
                        print(ex)
                    return
                if len(result_text) == 0:
                    await ctx.send("К сожалению, для этой картинки результатов нет")
                else:
                    await ctx.send("Вот, что удаось найти:")
                    for line in result_text:
                        await ctx.send(line)
                    await ctx.send("Обработка окончена!")
                try:
                    os.unlink(imageName)
                except Exception as ex:
                    print(ex)
            else:
                print("Это не картинка")
                await ctx.send("Это не картинка")
                return


@client.command()
async def translate(ctx):
    msg_content = str(ctx.message.content)
    if len(msg_content.rstrip()[10:]) == 0:
        await ctx.send("Укажи необходимые параметры. При необходимости обратись к translate_help")
        return
    lang = msg_content.rstrip()[10:].split()[0]
    qstn = " ".join(msg_content.rstrip()[10:].split()[1:])
    if qstn == "":
        await ctx.send("Напиши текст запроса")
        return
    try:
        translated = GoogleTranslator(source="auto", target=f"{lang}").translate(qstn)
    except Exception as ex:
        if str(ex) == f"{lang} --> There is no support for the chosen language":
            print("Данный язык не поддерживается")
            await ctx.send(f"Язык {lang} не поддерживается")
            return
        else:
            print(ex)
            await ctx.send("Что-то пошло не так при переводе")
            return
    if translated == "":
        await ctx.send("Что-то пошло не так при переводе")
        return
    else:
        await ctx.send("Результат перевода: \n")
        await ctx.send(translated)


client.run(TOKEN)
