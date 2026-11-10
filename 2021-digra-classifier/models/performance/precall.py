import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn.metrics import classification_report
onlyfiles = [f for f in listdir('models\performance') if isfile(join('models\performance',f))]

for file in onlyfiles:
    if file != 'precall.py':
        print(file)
        df = pd.read_csv('models\performance\\'+file)
        y_true = df.gold
        y_pred = df.preds
        print(classification_report(y_true, y_pred))