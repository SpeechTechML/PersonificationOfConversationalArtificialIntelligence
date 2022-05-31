import os
import sys 
import getopt

import pandas as pd


def parse_data(in_dir, out_path):

    # Finding files
    if in_dir.endswith('train'):
        dial_dir = os.path.join(in_dir, 'dialogues_train.txt')
        emo_dir = os.path.join(in_dir, 'dialogues_emotion_train.txt')
        act_dir = os.path.join(in_dir, 'dialogues_act_train.txt')
    elif in_dir.endswith('validation'):
        dial_dir = os.path.join(in_dir, 'dialogues_validation.txt')
        emo_dir = os.path.join(in_dir, 'dialogues_emotion_validation.txt')
        act_dir = os.path.join(in_dir, 'dialogues_act_validation.txt')
    elif in_dir.endswith('test'):
        dial_dir = os.path.join(in_dir, 'dialogues_test.txt')
        emo_dir = os.path.join(in_dir, 'dialogues_emotion_test.txt')
        act_dir = os.path.join(in_dir, 'dialogues_act_test.txt')
    else:
        print("Cannot find directory")
        sys.exit()

    # Open files
    in_dial = open(dial_dir, 'r')
    in_emo = open(emo_dir, 'r')
    in_act = open(act_dir, 'r')

    items = []
    for dia_id, (line_dial, line_emo, line_act) in enumerate(zip(in_dial, in_emo, in_act)):
        seqs = line_dial.split('__eou__')[:-1]
        emos = line_emo.split(' ')[:-1]
        acts = line_act.split(' ')[:-1]
        
        seq_len = len(seqs)
        emo_len = len(emos)
        act_len = len(acts)
        if seq_len != emo_len or seq_len != act_len:
            print("Different turns btw dialogue & emotion & acttion! ", dia_id + 1, seq_len, emo_len, act_len)
            sys.exit()

        for utt_id, (utt, emo, act) in enumerate(zip(seqs, emos, acts)):
            item = {
                'dia_id': dia_id,
                'utt_id': utt_id,
                'utterance': utt.strip(),
                'action': int(act),
                'emo': int(emo)
            }
            items.append(item)

    data_df = pd.DataFrame(items)
    data_df.to_csv(out_path)
    
    in_dial.close()
    in_emo.close()
    in_act.close()



def main(argv):

    in_dir = ''
    out_dir = ''

    try:
        opts, args = getopt.getopt(argv,"h:i:o:",["in_dir=","out_dir="])
    except getopt.GetoptError:
        print("python3 parser.py -i <in_dir> -o <out_dir>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("python3 parser.py -i <in_dir> -o <out_dir>")
            sys.exit()
        elif opt in ("-i", "--in_dir"):
            in_dir = arg
        elif opt in ("-o", "--out_dir"):
            out_dir = arg

    parse_data(in_dir, out_dir)

if __name__ == '__main__':
    main(sys.argv[1:])