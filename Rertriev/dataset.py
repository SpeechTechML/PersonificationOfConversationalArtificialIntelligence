import json
import csv
import random
import torch
import transformers
import numpy as np
def load_toloka(path):
    with open(path, 'r', encoding='utf-8') as data:
        for line in data:
            yield json.loads(line)
def tokenize(inp, tokenizer=False, max_len=32, join_token=False, type='gpt2'):
    '''
    tokenizer funk for PersonaChatTorchDataset and PersonaChatLazyDataset
    '''
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    if tokenizer:
        padding_side = tokenizer.padding_side
    if join_token:
        out=join_token.join(inp)
    else:
        out = inp 
    out = tokenizer(out, padding='max_length', max_length=max_len, truncation=True, return_tensors="pt")
    if type == 'bert':
        if padding_side == 'left':
            out = {k:out[k][:,-max_len:] for k in out}
        elif padding_side == 'right':
            out = {k:out[k][:,:max_len] for k in out}
        else:
            print('error')
        for k in out:
            cls_padder = torch.ones_like(out[k][:,:1])*cls_id
            out[k][:,:1] = torch.where((out[k][:,:1]!=pad_id), cls_padder, out[k][:,:1])
            out[k] = out[k].type(torch.IntTensor)
    elif type == 'bert_rcls':
        if type == 'bert':
            out = {k:out[k][:,-max_len:] for k in out}
        for k in out:
            cls_padder = torch.ones_like(out[k][:,-1:])*cls_id
            out[k][:,:1] = torch.where((out[k][:,-1:]!=pad_id), cls_padder, out[k][:,-1:])
            out[k] = out[k].type(torch.IntTensor)
    return out
class PersonaChatLazyDataset():
    def __init__(self, path, tokenizer_func=False, tokenizer=False, batch_size=32, context_len=32, responce_len=32, persona_len=32):
        self.path = path
        self.batch_size = batch_size
        self.tokenizer_func = tokenizer_func
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.responce_len = responce_len
        self.persona_len = persona_len
    def __iter__(self):
        with open(self.path, 'r', encoding='utf-8') as data:
            batch = None
            for line in data:
                x = json.loads(line)
                if batch is None:
                    batch = {k:[x[k]] for k in x}
                else:
                    for k in x:
                        batch[k].append(x[k])
                if len(batch['context']) == self.batch_size:
                    if self.tokenizer_func:
                        for k in batch:
                            if k == 'context':
                                max_len = self.context_len
                                join_token = self.tokenizer.end_of_masage_token
                            elif k == 'responce':
                                max_len = self.responce_len
                                join_token = False
                            elif k == 'persona':
                                max_len = self.persona_len
                                join_token = self.tokenizer.end_of_persona_sentence_token
                            else:
                                continue
                            batch[k] = self.tokenizer_func(batch[k], tokenizer=self.tokenizer, max_len=max_len, join_token=join_token)
                    yield batch, batch.pop('label')
                    batch = None
    def __next__(self):
        return json.loads(self.data.__next__())
    def __len__(self):
        c = 0
        for _ in self:
            c+=1
            print(c)
        return c
def clf(inp, tokenizer_func, tokenizer=False, context_len=32, responce_len=32, persona_len=32):
    '''
    collate_fn for PersonaChatTorchDataset.
    inp json lines [{context:[],
                    responce: str,
                    persona:[]}...]
    return batch {context:{inp_ids:[], token_type_ids:[], attention_mask:[]},
                  responce:{inp_ids:[], token_type_ids:[], attention_mask:[]},
                  persona:{inp_ids:[], token_type_ids:[], attention_mask:[]}}
    shape b_size:max_len
    return label [] shape b_size
    '''
    batch = None
    for line in inp:
        try:
            line = json.loads(line)
            if tokenizer_func:
                for k in line:
                    if k == 'context':
                        max_len = context_len
                        join_token = tokenizer.end_of_masage_token
                    elif k == 'responce':
                        max_len = responce_len
                        join_token = False
                    elif k == 'responce_aug':
                        line['responce'] = random.choice(line[k])
                        k = 'responce'
                        max_len = responce_len
                        join_token = False
                    elif k == 'persona':
                        max_len = persona_len
                        join_token = tokenizer.end_of_persona_sentence_token
                    elif k == 'persona_aug':
                        continue
                    else:
                        line[k] = [line[k]]
                        continue
                    tokens = tokenizer_func(line[k], tokenizer=tokenizer, max_len=max_len, join_token=join_token)
                    line[k] = {inp_type:tokens[inp_type][:32] for inp_type in tokens} #КОСТЫЛЬ
                try:
                    line.pop('responce_aug')
                    line.pop('persona_aug')
                except KeyError:
                    pass
                #print(line.keys())
                if batch is None:
                    batch = line
                else:
                    for k in line:
                        if k == 'label':
                            batch[k]+=line[k]
                        else:
                            for inp_type in line[k]:
                                batch[k][inp_type] = torch.cat((batch[k][inp_type], (line[k][inp_type])), 0)
        except Exception as e:
            print(e)
            pass
    return batch, batch.pop('label')

class PersonaChatTorchDataset(torch.utils.data.Dataset):
    def __init__(self, path): #, tokenizer_func=False, tokenizer=False, batch_size=32, context_len=32, responce_len=32, persona_len=32
        with open(path, 'r', encoding='utf-8') as data:
            self.data = data.readlines()
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
