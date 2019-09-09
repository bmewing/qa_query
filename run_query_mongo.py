import pymongo
import json
from bert_squad import QABot

MONGO_CLIENT = pymongo.MongoClient("mongodb://localhost:27017/")
DB = MONGO_CLIENT["ques_ans"]
COL = DB["nasdaq_ner_full_dbl"]
N_ANSWERS = 10

qa_bot = QABot(download=False)

while True:
    question = input("('exit' to quit) Question: ")
    if question.lower() == 'exit':
        break

    matches = COL.find({"$text": {"$search": question}},
                       {"score": {"$meta": "textScore"}}) \
        .sort([('score', {'$meta': 'textScore'})])\
        .limit(N_ANSWERS)

    max_score = -999
    best_answer = -999
    best_doc = {}
    for doc in matches:
        full_text = ' '.join(doc['text'])
        answer = qa_bot.ask_question(full_text, question)
        try:
            this_score = answer[2][0]
            if this_score > max_score:
                max_score = this_score
                best_answer = answer
                best_doc = doc
        except TypeError:
            this_score = -999

    full_text = ' '.join(best_doc['text'])
    answer_start_idx = best_answer[1][0]
    start = max([0, answer_start_idx - 20])
    end = min([len(full_text), answer_start_idx + 20])
    answer_context = '...' + full_text[start:end] + '...'

    final_answer = {
        'answer_text': best_answer[0][0],
        'answer_start_idx': answer_start_idx,
        'answer_context': answer_context,
        'answer_confidence': best_answer[2][0],
        'source_url': best_doc['url']
    }
    print(json.dumps(final_answer, indent=2))
