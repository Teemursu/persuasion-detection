import re
import pandas as pd
import os
import xmltodict, json, xmljson
from lxml.etree import fromstring, tostring
import lxml
from pprint import pprint
import xml.etree.ElementTree as ET
from nltk import word_tokenize
from pprint import pprint

dialogue_options = []

skills = []
game_col = []
filename = []
node_idx = []
df = pd.read_csv('nwn_kotor_persuade_checks_multilingual.csv')
persuade_lines = []
games = ["kotor2"]
perses = ["[Persu", ';Persu',
          "berreden]", "berreden&"
          "[Überre", ";Überreden", "[ Überre",
          '[Peru', ';Peru', "[ Persu", '[ Peru',
          '[Überreden/Lügen]', '[Überreden/Bluff]',]
for game in games:
    translations = {"game": [], "en": [], "de": [], "es": [], "fr": [], "it": []}
    str_idx = df['str_id'].loc[df['game'] == game].tolist()
    str_idx = list(map(int, str_idx))
    print(str_idx)
    for lang in translations.keys():
            if game != 'nwn' and lang == 'pl':
                pass
            else:
                with open(
                    "data_extraction\\games\\tlk_files\\{}_{}.txt".format(game, lang),
                    mode="r",
                    encoding="utf-8",
                ) as f:
                    lines = f.readlines()
                    for line in lines:
                        if any('string id="{}"'.format(str(str_id)) in line for str_id in str_idx):
                            #if len(line) > 110:
                              #line = line.split("]")
                              #print(line[1])
                              translations[lang].append(line)

                print(lang, game, len(translations[lang]))
    df = pd.DataFrame.from_dict(translations, orient="index").transpose()
    df.to_csv("{}_multilingual_persuade_lines.csv".format(game), mode="w", encoding='utf-8')
"""
with open("games\\tlk_files\\{}_dialog.xml".format(game), mode='r', encoding="utf-8", errors='ignore') as f:
            lines = f.readlines()
            for line in lines:
                #if re.search(r'\[Persuade', line) == False:
                if 'Persuade' not in line: 

                        #if "[Attack]" not in line and "[Leave]" not in line and "[Lie]" not in line:
                        #str_id = re.match(r'Persuasion', line)
                        #line = re.sub(r'<(.*?)>', "", line)
                        line = re.sub(r"\<\/string\>", "", line)
                        line = re.sub(r"\n", "", line)
                    #line = re.sub(r"\n", "", line)
                    # line = re.sub(r"\[", "", line)
                        line = line.split(">")
                        #print(line[1])
                        

                        try:
                            cleaned = re.sub(r"[\[\(\)\]\:\=\,\.]", "", line[1])
                            tokens = word_tokenize(cleaned)
                            #print("TEXT:",line[1])
                            if len(tokens) > 5:
                                persuade_lines.append(line[1])
                                str_id = line[0].split(">")
                                str_id =re.sub("<string id=", "", str_id[0])
                                node_idx.append(str_id)
                            
                            #print("ID:",str_id)
                        except IndexError:
                            #print("INDEXERROR",line)
                            pass

                        #line = line.split("]")
                        #dialogue_options.append((line, str_id[0], str_id[1]))
                        #skills.append(str_id[1])
df = pd.DataFrame(zip(node_idx, persuade_lines), columns=['str_id', 'lineence'])
df.to_csv(game+"en_nonpersuade_lines.csv",mode='wb',encoding='utf-8')
       
#pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in translation.items() ]))
"""
