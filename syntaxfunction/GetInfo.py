import json
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, Text
from collections import Counter
import sys
import getopt


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
                    conv += msgs
    result = []
    for i, conv in enumerate(dialogs):
        p1 = [rli(i) for i in conv[0].replace('your persona: ', '').split('\n') if rli(i)]
        p2 = [rli(i) for i in conv[1].replace(' partner\'s persona: ', '').split('\n') if rli(i)]
        dialog = [rli(i) for i in conv[2:]]
        for j in range(1, len(dialog)):
            context = [p for p in dialog[:j]]
            responce = dialog[j]
            if j % 2 == 0:
                persona = p2
            else:
                persona = p1

            result.append(json.dumps({'context': context, 'responce': responce, 'persona': persona, 'label': 1}) + '\n')
    return result


def main(argv):
    arg_input = ""
    arg_output = ""
    arg_help = "{0} -i <input> -o <output>".format(argv[0])
    try:
        opts, args = getopt.getopt(argv[1:], "hi:o:", ["help", "input=", "output="])
    except IndexError:
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-i", "--input"):
            arg_input = arg
        elif opt in ("-o", "--output"):
            arg_output = arg

    dialogs = bild_enpersonachat(arg_input)
    persons = []
    for i in range(len(dialogs)):
        persons_chr = []
        count_sent = 0
        count_rep = 0
        count_word = 0
        count_noun = 0
        count_verb = 0
        for dialog in dialogs:
            if json.loads(dialogs[i])["persona"] == json.loads(dialog)["persona"]:
                count_sent += len(sent_tokenize(str(json.loads(dialog)["responce"])))
                count_word += len(word_tokenize(str(json.loads(dialog)["responce"])))
                count_rep += 1
                tags = pos_tag(Text(word_tokenize(str(json.loads(dialog)["responce"]))))
                counts = Counter(tag for word, tag in tags)
                try:
                    count_noun += dict((word, count) for word, count in counts.items())["NN"]
                except (Exception,):
                    pass
                try:
                    count_verb += dict((word, count) for word, count in counts.items())["VB"]
                except (Exception,):
                    pass
                try:
                    count_verb += dict((word, count) for word, count in counts.items())["VBP"]
                except (Exception,):
                    pass
                try:
                    count_verb += dict((word, count) for word, count in counts.items())["VBD"]
                except (Exception,):
                    pass
                try:
                    count_verb += dict((word, count) for word, count in counts.items())["VBZ"]
                except (Exception,):
                    pass
                persons_chr.append(str(json.loads(dialog)["responce"]))
        persons.append({"name": str(i),
                        "sentence": count_sent / count_rep,
                        "word": count_word / count_rep,
                        "noun": count_noun / count_rep,
                        "verb": count_verb / count_rep})

    for person in persons:
        with open(arg_output, "a") as file:
            file.write(str(person).replace("'",'"') + "\n")
    return persons


if __name__ == "__main__":
    main(sys.argv)
