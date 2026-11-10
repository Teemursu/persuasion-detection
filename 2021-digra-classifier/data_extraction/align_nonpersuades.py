import pandas as pd
import re

kotor1 = pd.read_csv('kotor1_multilingual_persuade_lines.csv',index_col=[0])
kotor2 = pd.read_csv('kotor2_multilingual_persuade_lines.csv',index_col=[0])
nwn = pd.read_csv('nwn_multilingual_persuade_lines.csv',index_col=[0])

games_df = [kotor2]
games = ['kotor2']
str_idx = []
sentences = []

temp_dict = {"en": [], "de": [], "es": [], "fr": [], "it": []}

for game_name, game_df in zip(games, games_df):

    for lang in game_df.keys():
        for sent in game_df[lang]:
            sent = re.sub(' soundlength="0.000000"', "", str(sent))
            sent = re.sub("b'", "", str(sent))
            sent = sent.split('string id="')
            try:
                sent = sent[1].split("\"")
                str_id = sent[0]
                if game_name == 'kotor1':
                    sentence = sent[3]
                else:
                    sentence = sent[1]
                temp_dict[lang].append((str_id, sentence, game_name))
                if str_id not in str_idx:
                    str_idx.append(str_id)
            except IndexError:
                pass
        print("creating temp dict for",lang,"is done" )
            
translations_df = pd.DataFrame(columns = ['str_ids', 'game', 'en', 'de', 'es', 'fr', 'it',])
translations_df['str_ids'] = str_idx

for lang in temp_dict.keys():
    for elem in temp_dict[lang]:
        str_id = elem[0]
        sent = elem[1]
        sent = re.sub(">&lt;StartCheck&gt;", "", sent)
        sent = re.sub("&lt;/Start&gt;", "", sent)
        sent = re.sub("</string>", "", sent)
        game = elem[2]
        translations_df.loc[translations_df['str_ids'] == str_id, lang] = sent        
        #print()
        #print("ID:",str_id) 
        #print("Sent:", sent)
        #print("Game:", game)
    print("aligning",lang,"is done")
#translations_df = translations_df.dropna()
    translations_df.to_csv("{}_persuade_sents.csv".format(games[0]),mode='wb',encoding='utf-8')

    