import pandas as pd


def load_data(data_path):
    # To load corrected with Yandex.Speller .txt data
    if data_path[-3:] == 'txt':
        with open(data_path, 'r', encoding="utf-8") as infile:
            return infile.readlines()

    # To load raw data
    data = pd.read_csv(data_path, header=0, delimiter='\t')
    result = []

    for line in data.iterrows():
        ind = line[0]
        row = line[1]

        persona_1_profile = '|' + row['persona_1_profile'].replace("<span class=participant_1>", "")\
            .replace('</span>', '').replace('\r', '').replace('<br />', '|').replace('\n', '')\
            .replace('\"', '').replace('\\', '')[:-1]
        persona_1_profile = persona_1_profile[:len(persona_1_profile)]

        persona_2_profile = '|' + row['persona_2_profile'].replace("<span class=participant_2>", "")\
            .replace('</span>', '').replace('\r', '').replace('<br />', '|').replace('\n', '')\
            .replace('\"', '').replace('\\', '')[:-1]
        persona_2_profile = persona_2_profile[:len(persona_2_profile)]

        inp = row['dialogue'].replace("<span class=participant_1>", "")\
            .replace("<span class=participant_2>", "").replace('</span>', '').replace('\r', '')\
            .replace('<br />', '').replace('\n', '').replace('\"', '').replace('\\', '')

        if (inp.startswith("Пользователь 1")):
            result.append(persona_2_profile+'\n')
            result.append(persona_1_profile+'\n')
        else:
            result.append(persona_1_profile+'\n')
            result.append(persona_2_profile+'\n')
        result.append(inp+'\n')

    return result


def preprocess_data(config):
    data_path = config['data_path']
    max_length = config['max_length']
    persona_token = config['persona_token']
    you_token = config['you_token']
    other_token = config['other_token']
    is_causal_lm = config['is_causal_lm']

    labels = []
    context = []

    lines = load_data(data_path)
    lines = [line.rstrip() for line in lines]

    for i in range(0, len(lines), 3):
        persona1 = lines[i].replace('\n', '').replace('|', persona_token).replace('.', '')
        persona2 = lines[i+1].replace('\n', '').replace('|', persona_token).replace('.', '')

        dialogue = lines[i+2].replace("Пользователь 1: ", "\n<p1>").replace("Пользователь 2: ", "\n<p2>").split("\n")
        dialogue.pop(0)

        for i in range(1, len(dialogue) - 1):
            your_persona = ""
            persona_id = ""
            if dialogue[i+1][:4] == "<p1>":
                d_len = max_length*2 - len(persona1)
                your_persona = persona1
                persona_id = "<p1>"
            else:
                d_len = max_length*2 - len(persona2)
                your_persona = persona2
                persona_id = "<p2>"

            label = you_token + dialogue[i + 1][4:]
            dialogue_history = ""

            for j in range(i, 0, -1):
                if len(dialogue[j][4:] + dialogue_history) <= d_len:
                    if dialogue[j][:4] == persona_id:
                        dialogue_history = you_token + dialogue[j][4:] + dialogue_history
                    else:
                        dialogue_history = other_token + dialogue[j][4:] + dialogue_history
                else:
                    break

            if dialogue_history != "":
                # '<s>' only for CausalLM models
                if is_causal_lm:
                    context.append('<s>' + your_persona + dialogue_history)
                else:
                    context.append(your_persona + dialogue_history)
                labels.append(dialogue_history)

    return {
        "context": context,
        "labels": labels
        }
