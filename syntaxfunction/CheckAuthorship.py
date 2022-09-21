from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, Text
import json
from collections import Counter
import sys
import getopt
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def main(argv):
    replica = ""
    persons_info_path = ""
    arg_help = "{0} -r <replica> -p <persons_info_path>".format(argv[0])
    try:
        opts, args = getopt.getopt(argv[1:], "hr:p:", ["help", "replica=", "persons_info_path="])
    except IndexError:
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-r", "--replica"):
            replica = arg
        elif opt in ("-p", "--persons_info_path"):
            persons_info_path = arg
    count_word = len(word_tokenize(replica))
    count_sent = len(sent_tokenize(replica))
    tags = pos_tag(Text(word_tokenize(replica)))
    counts = Counter(tag for word, tag in tags)
    count_noun = 0
    count_verb = 0
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
    with open(persons_info_path, "r") as file:
        sum_diff = 1000
        name = ""
        for line in file:
            sum_futures = abs(count_word - int(json.loads(line)['word'])) \
                  + abs(count_sent - int(json.loads(line)['sentence'])) \
                  + abs(count_noun - int(json.loads(line)['noun']))\
                  + abs(count_verb - int(json.loads(line)['verb']))
            if sum_futures < sum_diff:
                sum_diff = sum_futures
                name = str(json.loads(line)["name"])
    print(f'This replica looks like {name} person')
    return f'This replica looks like {name} person'


if __name__ == "__main__":
    main(sys.argv)
