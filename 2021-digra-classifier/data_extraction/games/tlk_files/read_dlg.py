import re
import pandas as pd
import os
import xmltodict, json, xmljson
from lxml.etree import fromstring, tostring
import lxml
from pprint import pprint
import xml.etree.ElementTree as ET

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

difficulty = []
sentence = []
attribute = []
game = []
import re
game_str = 'Planescape Torment'
filenames = getListOfFiles('d_files/planescape_torment/')
for file in filenames:
    with open(file, errors="ignore") as f:
        lines = f.readlines()
        for line in lines:
            if "CheckStatGT" in line:
                attributes = re.search(r' \d\d, [A-Z]{3}', line)
                text = re.search('\/\* ~"(.*?)"', line)
                if attributes and text:
                    attributes = attributes.group(0)
                    text = text.group(0)
                    attributes = attributes.split(",")
                    text = re.sub("\/\* ~", "", text)
                    print(attributes, text)
                    difficulty.append(attributes[0])
                    attribute.append(attributes[1])
                    sentence.append(text)
                    game.append(file)
 
print(len(sentence))                   
df = pd.DataFrame(list(zip(game, sentence, attribute, difficulty)), columns=['game', 'sentence','attribute','difficulty'])
with open("planescape_attribute_requirements.csv", "w") as f:
    df.to_csv("planescape_attribute_requirements.csv")
                    