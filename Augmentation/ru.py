from pycorenlp import StanfordCoreNLP
import json
import re
import OpenAttack
import numpy as np
import os
import csv

script_path = os.path.dirname(__file__)
attacker = OpenAttack.attackers.SCPNAttacker()
nlp = StanfordCoreNLP('http://localhost:9000')

# CoreNlp Russian https://github.com/MANASLU8/CoreNLP.git


def get_dialog(inp, mod):
    if inp[0] == '"':
        inp = inp[1:]

    inp = inp.replace('<br />', ' ').replace('\r', ' ').replace('\n', ' ').split('</span> ')
    out = [inp[0]]
    for i in inp[1:]:
        if i[:42] == out[-1][:42]:
            out[-1] = out[-1] + '. ' + i[42:]
        elif len(i) > 25:
            out.append(i)
    return out


def bild_rupersonachat(raw):
    with open(raw, 'r', encoding='utf-8') as data:
        data = csv.reader(data, delimiter='\t')
        result = []
        for i, conv in enumerate(data):
            if i == 0:
                continue
            p1 = conv[0].replace('<span class=participant_1>', '')
            p2 = conv[1].replace('<span class=participant_2>', '')
            dialog = get_dialog(conv[2][:-1], 'join')
            for i in range(1, len(dialog)):
                context = [p[42:] for p in dialog[:i]]
                try:
                    persona, responce = dialog[i].replace('<span class=participant_1>Пользователь 1: ',
                                                          p1 + '[-sep-]') \
                        .replace('<span class=participant_2>Пользователь 2: ', p2 + '[-sep-]').split('[-sep-]')
                except BaseException:
                    print(conv)
                    C = 'a'
                    break

                persona = persona.replace('</span>', '').split('<br />')[:-1]
                result.append(json.dumps({'context': context, 'responce': responce, 'persona': persona, 'label': 1}))
    return result


def GetConsistuencyTemplate(sentence):
    get_parse = nlp.annotate(sentence, properties={'annotators': 'parse', 'outputFormat': 'json'})
    consistuency_tree = get_parse['sentences'][0]['parse']
    tree_template = ' '.join([word for word in ((re.sub('\s+', " ", consistuency_tree)).replace(")", " )")).split(' ') if (word != "I" and (not word.islower()))])
    return tree_template + " EOP"


def GetAllTemplates(dialogs):
    all_templates = []
    i = 0
    for i in range(len(dialogs)):
        all_templates.append(GetConsistuencyTemplate(json.loads(dialogs[i])['responce']))
    return all_templates


def SaveReplicsNumberByPerson(dialogs):
    persons = {}
    for i in range(len(dialogs)):
        values = []
        print(i)
        for j in range(len(dialogs)):
            if j != i:
                try:
                    if json.loads(dialogs[i])['persona'] == json.loads(dialogs[j])['persona']:
                        values.append(j)
                except (Exception, ):
                    pass
        persons[i] = values
    np.save('persons_ru.npy', persons)


def main():
    dialogs = bild_rupersonachat(f'{script_path}/tolokaPersonachat/dialogues.tsv')
    if os.path.isfile("persons.npy"):
        number_dictionary = np.load('persons_ru.npy', allow_pickle='TRUE').item()
    else:
        SaveReplicsNumberByPerson(dialogs)
        number_dictionary = np.load('persons_ru.npy', allow_pickle='TRUE').item()

    templates = GetAllTemplates(dialogs)
    i = 0
    for i in range(len(dialogs)):
        responce_templates = []
        for number in number_dictionary[i]:
            responce_templates.append((templates[number].replace("(", "( ")).replace("?", ""))
        if len(responce_templates) >= 5:
            responce_templates = responce_templates[0:4]
        dialog_line = json.loads(dialogs[i])
        try:
            dialog_line['responce_aug'] = attacker.gen_paraphrase(dialog_line['responce'], responce_templates)
        except:
            pass
        sorted_json = json.dumps(dialog_line, sort_keys=True)
        check_file = os.path.exists(f'{script_path}/aug_toloka/russian_aug.json')
        if check_file is False:
            os.mkdir(f'{script_path}/aug_toloka')
        with open("aug_toloka/russian_aug.json", 'a') as result:
            result.write(sorted_json + '\n')


if __name__ == "__main__":
    main()
