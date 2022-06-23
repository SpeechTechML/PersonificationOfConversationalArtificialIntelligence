from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, Text
import json
from collections import Counter

def checkAuthorship(replica, dialogs):
    count_word = len(word_tokenize(replica))
    count_sent = len(sent_tokenize(replica))
    tags = pos_tag(Text(word_tokenize(replica)))
    counts = Counter(tag for word, tag in tags)
    count_noun = 0
    count_verb = 0
    try:
        dict((word, count) for word, count in counts.items())["NN"]
        count_noun += dict((word, count) for word, count in counts.items())["NN"]
    except:
        pass
    try:
        dict((word, count) for word, count in counts.items())["VB"]
        count_verb += dict((word, count) for word, count in counts.items())["VB"]
    except:
        pass
    try:
        dict((word, count) for word, count in counts.items())["VBP"]
        count_verb += dict((word, count) for word, count in counts.items())["VBP"]
    except:
        pass
    try:
        dict((word, count) for word, count in counts.items())["VBD"]
        count_verb += dict((word, count) for word, count in counts.items())["VBD"]
    except:
        pass
    try:
        dict((word, count) for word, count in counts.items())["VBZ"]
        count_verb += dict((word, count) for word, count in counts.items())["VBZ"]
    except:
        pass
    with open(dialogs, "r") as file:
        sum_diff = 1000
        name = ""
        for line in file:
            sum = abs(count_word - int(json.loads(line)['word'])) \
                  + abs(count_sent - int(json.loads(line)['sentence'])) \
                  + abs(count_noun - int(json.loads(line)['noun']))\
                  + abs(count_verb - int(json.loads(line)['verb']))
            if sum < sum_diff:
                sum_diff = sum
                name = str(json.loads(line)["name"])
    print("This replica looks like %s person" % name)
    print(sum_diff)
checkAuthorship("how are you ? being an old man , i am slowing down these days", "persons.json")