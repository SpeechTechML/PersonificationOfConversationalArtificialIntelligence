from pycorenlp import StanfordCoreNLP
import json
import re
import OpenAttack
import numpy as np
from thefuzz import process
import os

nlp = StanfordCoreNLP('http://localhost:9000')
attacker = OpenAttack.attackers.SCPNAttacker()


def GetConsistuencyTemplate(sentence):
    get_parse = nlp.annotate(sentence, properties={'annotators': 'parse', 'outputFormat': 'json'})
    consistuency_tree = json.loads(get_parse)['sentences'][0]['parse']
    tree_template = ' '.join([word for word in ((re.sub('\s+', " ", consistuency_tree)).replace(")", " )")).split(' ') if (word != "I" and (not word.islower()))])
    return tree_template + " EOP"


def GetAllTemplates(dialogs):
    all_templates = []
    i = 0
    for i in range(len(dialogs)):
        all_templates.append(GetConsistuencyTemplate(json.loads(dialogs[i])['responce']))
    return all_templates


def rli(inp):
    splt = inp.split(' ')
    if splt[0].isdigit():
        inp = ' '.join(splt[1:])
    return inp


def bild_enpersonachat(raw):
    with open(raw, 'r', encoding='utf-8') as f:
        dialogs = []
        for i in f:
            if i[:2] == '1 ':
                dialogs.append([i])
            else:
                conv = dialogs[-1]
                if (conv[-1].split(' ')[1:3] == i.split(' ')[1:3]) and (i.split(' ')[2] == 'persona:'):
                    conv[-1] = conv[-1] + i  # .split(': ')[1]
                else:
                    msgs = i.split('\t\t')[0].split('\t')
                    for i, msg in enumerate(msgs):
                        newmsg = msg.split(' ')
                    conv += msgs
    result = []
    for i, conv in enumerate(dialogs):
        p1 = [rli(i) for i in conv[0].replace('your persona: ', '').split('\n') if rli(i)]
        p2 = [rli(i) for i in conv[1].replace(' partner\'s persona: ', '').split('\n') if rli(i)]
        dialog = [rli(i) for i in conv[2:]]
        for i in range(1, len(dialog)):
            context = [p for p in dialog[:i]]
            responce = dialog[i]
            if i % 2 == 0:
                persona = p2
            else:
                persona = p1

            result.append(json.dumps({'context': context, 'responce': responce, 'persona': persona, 'label': 1}))
    return result


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
                except:
                    pass
        persons[i] = values
    np.save('persons.npy', persons)


dialogs = bild_enpersonachat("personachat/test_both_original.txt")
if os.path.isfile("persons.npy"):
    number_dictionary = np.load('persons.npy', allow_pickle='TRUE').item()
else:
    SaveReplicsNumberByPerson(dialogs)
    number_dictionary = np.load('persons.npy', allow_pickle='TRUE').item()

templates = GetAllTemplates(dialogs)
i = 0
for i in range(len(dialogs)):
    responce_templates = []
    for number in number_dictionary[i]:
        responce_templates.append(templates[number])
    responce_templates = process.dedupe(responce_templates, threshold=98)
    if len(responce_templates) >= 5:
        responce_templates = responce_templates[0:4]
    dialog_line = json.loads(dialogs[i])
    dialog_line['responce_aug'] = attacker.gen_paraphrase(dialog_line['responce'],
                                                          responce_templates)
    sorted_json = json.dumps(dialog_line, sort_keys=True)
    with open("aug_personachat/test_both_original_aug.json", 'a') as result:
        result.write(sorted_json + '\n')
