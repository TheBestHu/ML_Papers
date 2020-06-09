import glob
import sys

file_path = "d:\\ML_Papers\\TansferL\\"

with open(file_path + 'README.md', 'w', encoding='utf-8') as f:
    f.write("## Word Embedding \n")
    for file in glob.glob(file_path + "*.pdf"):
        file = file.split('\\')[-1].split('.')[0]
        f.write("- {} \n".format(file))