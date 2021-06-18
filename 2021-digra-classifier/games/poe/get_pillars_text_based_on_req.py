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


skills = []
lines = []
game = []
filename = []
node_idx = []
game = 'tyranny'
#dialogue_files = getListOfFiles('pillars_of_eternity/dialogue_text/')
#attribute_files = getListOfFiles('pillars_of_eternity/attribute_skill_requirements/')

with open(game+'_attribute_requirements.csv','r') as f:
    data = pd.read_csv(f)    
    data['text'] = ""

for i, file in enumerate(data['filename']):
    data['filename'][i] = re.sub('attribute_skill_requirements', 'dialogue_text', file)
    data['filename'][i] = re.sub('\\\\', '/', data['filename'][i])
    data['filename'][i] = re.sub('conversation', 'stringtable', data['filename'][i])
    
    try:
        tree = ET.parse(data['filename'][i])
        root = tree.getroot()
        for child in root.findall('Entries'):
            entries = child.findall('Entry')
            for entry in entries:
                identifier = int(entry.find('ID').text)
                attribute_ids = data['nodeid'].loc[data['filename'] == data['filename'][i]]
                text = entry.find('DefaultText').text
                
                for value in attribute_ids:
                    if value == identifier:
                            data['text'][i] = text
                        
                    # "If the island is cursed, who ever lived to tell of it?" 
                    # should be 29 of pillars_of_eternity/attribute_skill_requirements/00_prototype\00_cv_nairi.conversationbundle
                    # ('Intellect', '13')
                    
                    # pillars_of_eternity/dialogue_text/00_prototype/00_cv_tikawara_resolution.stringtable 4    64
                    # "Careful. I'm not someone you want to offend.
            
    except FileNotFoundError:
        print(file, "not found")
        
        
with open(game+"_persuasion_data.csv", "w") as f:
        data.to_csv(game+"_persuasion_data.csv")