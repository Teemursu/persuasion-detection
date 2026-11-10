import re
import pandas as pd
import os
import xmltodict, json, xmljson
from lxml.etree import fromstring, tostring
import lxml
from pprint import pprint
import xml.etree.ElementTree as ET

skills = []
lines = []
game = []
filename = []
node_idx = []
game = 'Planescape Torment'
dialogue_options = []

file = 'games\\tlk_files\\dialog.tlk'
with open(file, encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()
    for line in lines:
        if re.search(r'\[\w+\]', line):
                #f re.search(r'Resolve', line):
                    #if "[Attack]" not in line and "[Leave]" not in line and "[Lie]" not in line:
                        #line = re.sub(r'<string id="\d+">', "", line)
                    # line = re.sub(r"\<\/string\>", "", line)
                    # line = re.sub(r"\n", "", line)
                        #line = re.sub(r"\[", "", line)
                        print(line)
                        
                        line = line.split("]")
                        #print(line)
                        dialogue_options.append(line)


    for elem in dialogue_options:
        skills.append(elem[0])
        lines.append(elem[1])
        game.append(game)
        
df = pd.DataFrame(list(zip(game, lines, skills)), columns=['game', 'lines', 'attributes'])       
with open(game+"_persuasion_data.csv", "w") as f:
        df.to_csv(game+"_persuasion_data.csv")