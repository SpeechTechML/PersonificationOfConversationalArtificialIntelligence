import json
import csv
import torch
import transformers
import numpy as np
import tqdm
import os
from cobert import match, aggregate, fuse, dot_product_loss, train_epoch, evaluate_epoch
from dataset import PersonaChatTorchDataset, clf, tokenize
from util import logger
def run(train, val, models, lr, t_total, epochs, has_persona, gradient_accumulation_steps, device, fp16, 
        amp, apply_interaction, matching_method, aggregation_method, epoch_train_losses, epoch_valid_losses, 
        epoch_valid_accs, epoch_valid_recalls, epoch_valid_MRRs, best_model_statedict, writer=None, save_model_path=False, test_mode=False):
    optimizers = []
    schedulers = []
    for i, model in enumerate(models):
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
        optimizers.append(optimizer)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )
        schedulers.append(scheduler)
        
    for epoch in range(epochs):
        print("Epoch", epoch+1, '/', epochs)
        # training
        for model in models:
            model.train()
        train = tqdm.tqdm(train, desc="Iteration")
        train_loss, (train_acc, _, _) = train_epoch(data_iter=train, 
                                                    models=models, has_persona=has_persona, optimizers=optimizers, 
                                                    schedulers=schedulers, 
                                                    gradient_accumulation_steps=gradient_accumulation_steps, 
                                                    device=device, fp16=fp16, 
                                                    amp=amp, apply_interaction=apply_interaction, 
                                                    matching_method=matching_method, 
                                                    aggregation_method=aggregation_method)
        epoch_train_losses.append(train_loss)
        # evaluation
        for model in models:
            model.eval()
        valid_iterator = tqdm.tqdm(val, desc="Iteration")
        valid_loss, (valid_acc, valid_recall, valid_MRR) = evaluate_epoch(data_iter=val, models=models,
                                                                            has_persona=has_persona,
                                                                            gradient_accumulation_steps=gradient_accumulation_steps, 
                                                                            device=device, epoch=epoch, apply_interaction=apply_interaction, 
                                                                            matching_method=matching_method, aggregation_method=aggregation_method)
        print("Epoch {0}: train loss: {1:.4f}, valid loss: {2:.4f}, train_acc: {3:.4f}, valid acc: {4:.4f}, valid recall: {5}, valid_MRR: {6:.4f}"
            .format(epoch+1, train_loss, valid_loss, train_acc, valid_acc, valid_recall, valid_MRR))
        if writer is not None:
            writer.writerow({'epoch':epoch+1, 'train_loss':train_loss, 'valid_loss': valid_loss, 'train_acc':train_acc, 
                             'valid_acc':valid_acc, 'valid_r1':valid_recall[0], 'valid_r5':valid_recall[1], 'valid_r10':valid_recall[2], 'valid_MRR':valid_MRR})
        epoch_valid_losses.append(valid_loss)
        epoch_valid_accs.append(valid_acc)
        epoch_valid_recalls.append(valid_recall)
        epoch_valid_MRRs.append(valid_MRR)
        if save_model_path:
            if epoch == 0:
                for k, v in models.state_dict().items():
                    best_model_statedict[k] = v.cpu()
            else:
                if epoch_valid_recalls[-1][0] == max([recall1 for recall1, _, _ in epoch_valid_recalls]):
                    for k, v in models.state_dict().items():
                        best_model_statedict[k] = v.cpu()
                        
with open('config.json', 'r') as config:
    config  = json.loads(config.read())
save_model_path = config['save_model_path']
gradient_accumulation_steps = config['gradient_accumulation_steps']
matching_method = config['matching_method']
lr =  config['lr'] 
warmup_steps = config['warmup_steps']
test_mode = config['test_mode']
has_persona = config['has_persona']
context_len = config['context_len']
responce_len = config['responce_len']
persona_len = config['persona_len']
train_batch_size = config['train_batch_size']
val_batch_size = config['val_batch_size']
epochs = config['epochs']
split = config['split']
no_decay = ["bias", "LayerNorm.weight"]
fp16 = False
amp = None
weight_decay = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_path = config['model_path']
proc_data = config['proc_data_path']
apply_interaction = config['apply_interaction']
aggregation_method = config['aggregation_method']
padding_side = config['padding_side']
bert_config = transformers.BertConfig.from_pretrained(bert_path)
bert_tokenizer = transformers.BertTokenizer.from_pretrained(bert_path, padding_side=padding_side)
ctx_model = transformers.BertModel(bert_config).from_pretrained(bert_path)
cnd_model = transformers.BertModel(bert_config).from_pretrained(bert_path)
prs_model = transformers.BertModel(bert_config).from_pretrained(bert_path)
all_models = [ctx_model, cnd_model, prs_model]
data = PersonaChatTorchDataset(proc_data)
split = len(data)//config['split']
train, val = torch.utils.data.random_split(data, [len(data)-split, split])
train = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                        shuffle=True, num_workers=0, 
                        collate_fn=lambda x: clf(x, tokenizer_func=tokenize, 
                                                 tokenizer=bert_tokenizer, 
                                                 context_len=context_len, 
                                                 responce_len=responce_len, 
                                                 persona_len=persona_len))
val = torch.utils.data.DataLoader(val, batch_size=val_batch_size,
                        shuffle=True, num_workers=0, 
                        collate_fn=lambda x: clf(x, tokenizer_func=tokenize, 
                                                 tokenizer=bert_tokenizer, 
                                                 context_len=context_len, 
                                                 responce_len=responce_len, 
                                                 persona_len=persona_len))
print('\ntrain:', len(train), 'val:', len(val))
t_total = len(train) // gradient_accumulation_steps * train_batch_size
log_path = '(5-10)'+bert_path.split('/')[-1] + '_' + proc_data.split('/')[-1].split('.')[0] + '_interaction' + str(apply_interaction) +'_' + aggregation_method + '.csv'
log_path = 'logs/'+log_path
print(log_path)
epoch_train_losses = []
epoch_valid_losses = []
epoch_valid_accs = []
epoch_valid_recalls = []
epoch_valid_MRRs = []
best_model_statedict = {}
                    
with open(log_path, 'w') as log:
    writer = csv.DictWriter(log, fieldnames=['epoch', 'train_loss', 'valid_loss', 'train_acc', 'valid_acc', 'valid_r1', 'valid_r5', 'valid_r10', 'valid_MRR'],
                            delimiter=';', 
                            quotechar='"')
    writer.writeheader()

    models = all_models[:1]
    run(train, val, models, lr, t_total, 5, has_persona, gradient_accumulation_steps, device, fp16, 
        amp, apply_interaction, matching_method, aggregation_method, epoch_train_losses, epoch_valid_losses, 
        epoch_valid_accs, epoch_valid_recalls, epoch_valid_MRRs, best_model_statedict, writer, save_model_path, test_mode)

    if has_persona:
        [m.load_state_dict(models[0].state_dict()) for m in  all_models]
        run(train, val, all_models, lr/10, t_total, 10, has_persona, gradient_accumulation_steps, device, fp16, 
            amp, apply_interaction, matching_method, aggregation_method, epoch_train_losses, epoch_valid_losses, 
            epoch_valid_accs, epoch_valid_recalls, epoch_valid_MRRs, best_model_statedict, writer, save_model_path, test_mode)
