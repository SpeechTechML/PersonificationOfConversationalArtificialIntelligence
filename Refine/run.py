import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM

from Refine.data.data_loader import preprocess_data
from Refine.dataset import TextDataset, collate_fn
from Refine.train import train_model, eval_model, config_training


def init_config(model_name, is_causal_lm, max_length):
    # models = ["t5-large", "t5-base", "mt",
    # "t5-base-trained", "mt-trained", "dialogpt3",]

    config = {"data_path": "/data/toloka_speller.txt",
              "save_path": "/models/" + model_name + "-trained",
              "model": "/models/" + model_name,
              "is_causal_lm": is_causal_lm,
              "you_token": "<you>",
              "other_token": "<oth>",
              "persona_token": "<per>",
              "max_length": max_length,
              "batch_size": 8,
              "layers": None,
              "num_epochs": 5,
              "lr": 5e-5,
              "num_warmup_steps": 500,
              "accumulation_steps": 2,
              }

    return config


def save_model(save_path, model, tokenizer):
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def init_model(config):
    path = config['model']
    max_length = config['max_length']
    ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens': [config['you_token'], config['other_token'], config['persona_token']]}

    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    if config['is_causal_lm']:
        model = AutoModelForCausalLM.from_pretrained(path, max_length=max_length, output_attentions=True)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(path, max_length=max_length, output_attentions=True)

    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)

    model.resize_token_embeddings(new_num_tokens=model.config.vocab_size + num_added_tokens)

    # Only for MatianMT pretrained model
    if config['model'][8:] == 'mt':
        model.target_vocab_size = model.config.vocab_size

    return model, tokenizer


def run(model_name="t5-base", is_causal_lm=False, max_length=128):
    config = init_config(model_name, is_causal_lm, max_length)

    model, tokenizer = init_model(config)

    dataset = preprocess_data(config)

    dataset_size = len(dataset['labels'])

    train_size = int(0.8 * dataset_size)
    eval_size = dataset_size - train_size

    data = TextDataset(dataset)
    train, val = torch.utils.data.random_split(
        data,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = torch.utils.data.DataLoader(train, batch_size=config['batch_size'],
                                               shuffle=True, num_workers=0,
                                               collate_fn=collate_fn(config, train, tokenizer))
    val_loader = torch.utils.data.DataLoader(val, batch_size=config['batch_size'],
                                             shuffle=True, num_workers=0,
                                             collate_fn=collate_fn(config, val, tokenizer)
                                             )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)

    optimizer, lr_scheduler = config_training(model, train_loader,
                                              config['layers'], lr=config['lr'],
                                              num_epochs=config['num_epochs'],
                                              num_warmup_steps=config['num_warmup_steps'])

    train_model(model, tokenizer, train_loader, val_loader,
                config['num_epochs'], optimizer, lr_scheduler,
                device, accumulation_steps=config['accumulation_steps']
                )

    save_model(config['save_path'], model, tokenizer)
