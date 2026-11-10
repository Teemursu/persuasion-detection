import re
import pandas as pd
import os
import xmltodict, json, xmljson
from lxml.etree import fromstring, tostring
import lxml
from pprint import pprint

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
difficulty = []
game = 'tyranny'
dialogue_files = getListOfFiles('tyranny/dialogue_text/')
attribute_files = getListOfFiles('tyranny/attribute_skill_requirements/')
for file in attribute_files:
    #print(file)
    with open(file, 'rb') as f:
        o = xmltodict.parse(f)
        lines = json.dumps(o)
        # lines = json.load(f)
        #lines = lxml.etree.XML(lines)
        #xml = fromstring(str(lines))
        #print(xml)
        #json = json.dumps(lines)
        if game == 'poe2':
            for i in range(len(lines['Conversations'][0]['Nodes'])):
                if len(lines['Conversations'][0]['Nodes'][i]['Conditionals']['Components']) > 0:
                    try:
                        if 'IsAttributeScoreValue' in lines['Conversations'][0]['Nodes'][i]['Conditionals']['Components'][0]['Data']['FullName']:
                                pprint(lines['Conversations'][0]['Nodes'][i]['Conditionals']['Components'][0]['Data']['Parameters'])
                                skills.append(lines['Conversations'][0]['Nodes'][i]['Conditionals']['Components'][0]['Data']['Parameters'][1])
                                difficulty.append(lines['Conversations'][0]['Nodes'][i]['Conditionals']['Components'][0]['Data']['Parameters'][3])
                                filename.append(file)
                                node_idx.append(lines['Conversations'][0]['Nodes'][i]['NodeID'])
                    except KeyError:
                                pass 
                            
        if game == 'poe1':
            lines = json.loads(lines)
            for i in range(len(lines['ConversationData']['Nodes']['FlowChartNode'])):
                try:
                    try:
                        if 'Boolean IsPlayerAttributeScoreValue(AttributeScoreType, Operator, Int32)' in lines['ConversationData']['Nodes']['FlowChartNode'][i]['Conditionals']['Components']['ExpressionComponent']['Data']['FullName']:
                            
                            skills.append(lines['ConversationData']['Nodes']['FlowChartNode'][i]['Conditionals']['Components']['ExpressionComponent']['Data']['Parameters']['string'][0])
                            difficulty.append(lines['ConversationData']['Nodes']['FlowChartNode'][i]['Conditionals']['Components']['ExpressionComponent']['Data']['Parameters']['string'][2])
                            node_idx.append(lines['ConversationData']['Nodes']['FlowChartNode'][i]['NodeID'])
                            filename.append(file)
                    except TypeError:
                        pass
                except KeyError:
                    pass

        if game == 'tyranny':
            lines = json.loads(lines)
            for i in range(len(lines['ConversationData']['Nodes']['FlowChartNode'])):
                try:
                    try:
                        if 'Boolean IsPlayerScaledSkillValue(Guid, SkillType, Difficulty, Operator)' in lines['ConversationData']['Nodes']['FlowChartNode'][i]['Conditionals']['Components']['ExpressionComponent']['Data']['FullName']:
                            try:
                                skills.append(lines['ConversationData']['Nodes']['FlowChartNode'][i]['Conditionals']['Components']['ExpressionComponent']['Data']['Parameters']['string'][1])
                                difficulty.append(lines['ConversationData']['Nodes']['FlowChartNode'][i]['Conditionals']['Components']['ExpressionComponent']['Data']['Parameters']['string'][2])
                                node_idx.append(lines['ConversationData']['Nodes']['FlowChartNode'][i]['NodeID'])
                                filename.append(file)
                            except IndexError:
                                pass
                    except TypeError:
                        pass
                except KeyError:
                    pass
df = pd.DataFrame(list(zip(filename, node_idx, skills, difficulty)), columns=['filename', 'nodeid', 'attributes', 'difficulty'])
with open(game+"_attribute_requirements.csv", "w") as f:
        df.to_csv(game+"_attribute_requirements.csv")