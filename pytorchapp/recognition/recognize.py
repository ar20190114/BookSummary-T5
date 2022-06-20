from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
import re
import neologdn
from numpy import column_stack
import pandas as pd
import unicodedata
import tarfile
import random
from tqdm import tqdm
import argparse
import glob
import os
import time
import logging
from itertools import chain
from string import punctuation
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import textwrap
from tqdm.auto import tqdm
from sklearn import metrics
import io
import pandas as pd
import numpy as np
import pickle, gzip, urllib, json
import csv

import openpyxl
import time
import requests
from bs4 import BeautifulSoup
import json
import re
import neologdn
from numpy import column_stack
import pandas as pd
import pathlib
from selenium import webdriver


def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s


def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s


def normalize_neologd(s):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]+', '〜', s)  # normalize tildes (modified by Isao Sonobe)
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s


def normalize_text(text):
    assert "\n" not in text and "\r" not in text
    text = text.replace("\t", " ")
    text = text.strip()
    text = normalize_neologd(text)
    text = text.lower()
    return text


def preprocess_body(text):
    return normalize_text(text.replace("\n", " "))


#データの正規化
def preprocessText(text):

    text = re.sub(r'[\r\t\n\u3000]', '', text)
    text = neologdn.normalize(text)
    return text


def main(Title):
    PRETRAINED_MODEL_NAME = 'sonoisa/t5-base-japanese'

    # 各種ハイパーパラメータ
    args_dict = dict(
        data_dir="data",  # データセットのディレクトリ
        model_name_or_path=PRETRAINED_MODEL_NAME,
        tokenizer_name_or_path=PRETRAINED_MODEL_NAME,
        
        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        gradient_accumulation_steps=1,

        early_stop_callback=False,
        fp_16=False,
        opt_level='O1',
        max_grad_norm=1.0,
        seed=42,
    )

    args_dict.update({
        "max_input_length":  512,  # 入力文の最大トークン数
        "max_target_length": 512,  # 出力文の最大トークン数
        "train_batch_size":  2,
        "eval_batch_size":   2,
        "num_train_epochs":  1,
        })

    args = argparse.Namespace(**args_dict)


    MODEL_DIR = './model1'

    # トークナイザー（SentencePiece）
    tokenizer = T5Tokenizer.from_pretrained("ryota/newsCreate", is_fast=True, use_auth_token=True)

    # 学習済みモデル
    trained_model = T5ForConditionalGeneration.from_pretrained("ryota/newsCreate", use_auth_token=True)

    MAX_SOURCE_LENGTH = args.max_input_length   # 入力される記事本文の最大トークン数
    MAX_TARGET_LENGTH = args.max_target_length  # 生成されるタイトルの最大トークン数


    # 推論モード設定
    trained_model.eval()




    # googleで検索する文字
    search_string = Title + '青空文庫'

    #Seleniumを使うための設定とgoogleの画面への遷移
    INTERVAL = 2.5
    URL = "https://www.google.com/"
    driver_path = "./chromedriver"
    driver = webdriver.Chrome(executable_path=driver_path)
    driver.maximize_window()
    time.sleep(INTERVAL)
    driver.get(URL)
    time.sleep(INTERVAL)

    #文字を入力して検索
    driver.find_element_by_name('q').send_keys(search_string)
    driver.find_elements_by_name('btnK')[1].click() #btnKが2つあるので、その内の後の方
    time.sleep(INTERVAL)

    #検索結果の一覧を取得する
    results = []
    flag = False
    while True:
        g_ary = driver.find_elements_by_class_name('g')
        for g in g_ary:
            result = {}
            result['url'] = g.find_element_by_class_name('yuRUbf').find_element_by_tag_name('a').get_attribute('href')
            result['title'] = g.find_element_by_tag_name('h3').text
            results.append(result)
            if len(results) >= 1: #抽出する件数を指定
                flag = True
                break
        if flag:
            break
        driver.find_element_by_id('pnnext').click()
        time.sleep(INTERVAL)

    #ワークブックの作成とヘッダ入力
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet['A1'].value = 'タイトル'
    sheet['B1'].value = 'URL'

    #シートにタイトルとURLの書き込み
    for row, result in enumerate(results, 2):
        sheet[f"A{row}"] = result['title']
        sheet[f"B{row}"] = result['url']


    # スクレイピング
    # Webページを取得して解析する
    load_url = result['url']
    html = requests.get(load_url)
    soup = BeautifulSoup(html.content, "html.parser")

    Text = ''

    for element in soup.find_all('div', class_='main_text'):
        Text = Text + element.text  # すべてのliタグを検索して表示
        # print(element.text)

    body = preprocessText(Text)
    print(body)

    # 前処理とトークナイズを行う
    inputs = [preprocess_body(body)]
    batch = tokenizer.batch_encode_plus(
        inputs, max_length=MAX_SOURCE_LENGTH, truncation=True, 
        padding="longest", return_tensors="pt")

    input_ids = batch['input_ids']
    input_mask = batch['attention_mask']

    # 生成処理を行う
    outputs = trained_model.generate(
        input_ids=input_ids, attention_mask=input_mask, 
        max_length=MAX_TARGET_LENGTH,
        temperature=1.0,          # 生成にランダム性を入れる温度パラメータ
        num_beams=10,             # ビームサーチの探索幅
        diversity_penalty=1.0,    # 生成結果の多様性を生み出すためのペナルティ
        num_beam_groups=10,       # ビームサーチのグループ数
        num_return_sequences=1,  # 生成する文の数
        repetition_penalty=1.5,   # 同じ文の繰り返し（モード崩壊）へのペナルティ
    )

    # 生成されたトークン列を文字列に変換する
    generated_titles = [tokenizer.decode(ids, skip_special_tokens=True, 
                                        clean_up_tokenization_spaces=False) 
                        for ids in outputs]

    # 生成されたタイトルを表示する
    Title_body = []
    for i, title in enumerate(generated_titles):
        Title_body.append(title)
        print(f"{i+1:2}. {title}")

    return Title_body
        

if __name__ == '__main__':
    main()