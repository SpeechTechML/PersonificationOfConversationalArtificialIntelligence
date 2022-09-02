import numpy as np
import torch
from torch.optim import AdamW
from torchtext.data.metrics import bleu_score
from transformers import get_scheduler
from sklearn.metrics import f1_score, recall_score
from tqdm.notebook import tqdm


def config_training(model, train_loader, layers,
                    lr, num_epochs, num_warmup_steps):
    if layers != None:
        for i, param in model.named_parameters():
            param.requires_grad = False
            for layer in layers:
                if layer in i and param.requires_grad == False:
                    param.requires_grad = True
                    print(f"{i}:   {param.shape}")

    optimizer = AdamW(model.parameters(), lr=lr)

    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    return optimizer, lr_scheduler


def compute_bleu_f1(model, tokenizer, inputs, labels):
    results = {'bleu': [],
               'f1': [],
               'recall': []}

    preds = model.generate(
        **inputs,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
        no_repeat_ngram_size=3
    ).to('cpu').detach().numpy()

    preds = list(preds)
    str_preds = []
    str_labels = []
    for i in range(len(preds)):
        line = []
        for j in range(len(preds[i])):
            line.append(str(preds[i][j]))
        str_preds.append(line)

    for i in range(len(labels)):
        line = []
        for j in range(len(labels[i])):
            line.append(str(labels[i][j]))
        str_labels.append(line)
        while '-100' in str_labels[i]:
            str_labels[i].remove('-100')

    for pred, label in zip(str_preds, str_labels):
        if len(label) < len(pred):
            label.extend(['-100' for _ in range(len(label), len(pred))])
        elif len(label) > len(pred):
            pred.extend([str(tokenizer.pad_token_id) for _ in range(len(pred), len(label))])

        results['bleu'].append(bleu_score([pred], [[label]], max_n=4, weights=[0.25, 0.25, 0.25, 0.25]))
        results['f1'].append(f1_score(label, pred, average='macro'))
        results['recall'].append(recall_score(label, pred, average='macro'))
    return results


def eval_model(model, tokenizer, val_loader,
               device, skip_generation=False):
    print("Validation")
    losses = []
    bleu = []
    f1 = []
    recall = []

    progress_bar = tqdm(range(len(val_loader)))

    model.eval()
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits.to('cpu')

        labels = batch.pop('labels').to('cpu').detach().numpy()

        if not skip_generation:
            results = compute_bleu_f1(model, tokenizer, batch, labels)
            bleu.extend(results['bleu'])
            f1.extend(results['f1'])
            recall.extend(results['recall'])

        loss = outputs.loss
        losses.append(loss.to('cpu').detach().numpy())

        progress_bar.update(1)

    print(f"val loss: {np.mean(losses)}")
    print(f"val ppl: {np.exp(np.mean(losses))}")
    if not skip_generation:
        print(f"val bleu: {np.mean(bleu)}")
        print(f"val f1: {np.mean(f1)}")
        print(f"val R@1: {np.mean(recall)}")


def train_model(model, tokenizer, train_loader, val_loader,
                num_epochs, optimizer, lr_scheduler,
                device, accumulation_steps=1):
    ppl = []

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}")
        print("Training")
        progress_bar = tqdm(range(len(train_loader)))
        optimizer.zero_grad()

        i = 1
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / accumulation_steps
            loss.backward()

            if i % accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            i += 1

        eval_model(model, tokenizer, val_loader, skip_generation=True)
