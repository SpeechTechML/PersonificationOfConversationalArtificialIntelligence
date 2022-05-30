import torch


def generate_answers(model, tokenizer):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)

    persona = ["Я программист",
            "Живу в Ростове",
            "Не умею играть на гитаре",
            "Обожаю готовить",
            "Не люблю кошек"]

    personas = ["<per>Я программист<per>Живу в Ростове<per>Не умею играть на гитаре<per>Работаю два дня<per>Не люблю кошек",
                "<per>Я домохозяйка<per>Вышла замуж после школы<per>Есть двое детей<per>Обожаю готовить<per>Мечтаю об отпуске"]

    utterance = "Ты умеешь играть на гитаре?"

    for i in persona:
        print(i)
    print("________________________________________")
    print(utterance + '\n')
    outs = []

    context = "<per>" + "<per>".join(persona) + "<oth>" + utterance

    inputs = tokenizer.encode(
        context,
        return_tensors="pt"
        ).to(device)
    for i in range(5):
        outs.append(model.generate(
            inputs,
            #min_length=5,
            #max_length=25,
            do_sample=True,
            temperature=1.0,
            top_k=100,
            no_repeat_ngram_size=3,
            num_return_sequences=2
            ))
    for i in range(5):
        for j in range(2):
            print(f"Candidate {i*2 + j + 1}: " + tokenizer.decode(outs[i][j], skip_special_tokens=True))


def start_dialogue(model, tokenizer, other_token, you_token):
    persona = "<per>Я программист<per>Люблю кошек<per>Есть двое детей<per>Работаю два дня<per>Ем пиццу"
    dialogue_session = persona

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    stop_word = "stop"
    inp = input("In: ")
    while inp != stop_word:
        dialogue_session += other_token + inp
        inputs = tokenizer.encode(dialogue_session, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs,
            #min_length=5,
            #max_length=25,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            no_repeat_ngram_size=3,
            num_return_sequences=1
        )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Out: " + decoded + "\n")
        dialogue_session += you_token + decoded

        inp = input("In: ")