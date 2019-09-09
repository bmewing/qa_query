import pymongo
import glob
import json
import re
import nltk
from tqdm import tqdm

MONGO_CLIENT = pymongo.MongoClient("mongodb://localhost:27017/")
DB = MONGO_CLIENT["ques_ans"]
COL = DB["nasdaq_ner_full_dbl"]
COL.drop()


def nltk_ner_extract(text, keep_all=True, join=False):
    filtered_ner = ['ORGANIZATION', 'LOCATION', 'FACILITY', 'GPE']

    chunks = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
    ne_list = []
    for chunk in chunks:
        if hasattr(chunk, 'label'):
            if keep_all or chunk.label() in filtered_ner:
                full_ne = ' '.join(c[0] for c in chunk)
                ne_list.append(full_ne)

    if join:
        return ' '.join(ne_list)
    else:
        return ne_list


def load_json_articles_to_db(collection, json_glob='data/articles/*.json', ner=True):
    for p in tqdm(glob.glob(json_glob)):
        with open(p, 'r') as f:
            article = json.load(f)
            if ner:
                article['ner'] = nltk_ner_extract(' '.join(article['text']))
            article['text'] = re.sub('   +', '\n', ''.join(article['text'])).split('\n')
            collection.insert_one(article)


def create_index(collection, column):
    collection.create_index(column, default_language='english')


if __name__ == '__main__':
    load_json_articles_to_db(COL)
    create_index(COL, [('ner', pymongo.TEXT), ('text', pymongo.TEXT)])
