import torch


def collate_fn(config, data, tokenizer):
    texts, labels = zip(*data)
    
    max_length = config['max_length']
    other_token = 50258
    you_token = 50257
    bos_token = 1
    eos_token = 2
    
    labels = tokenizer(list(labels), max_length=max_length, truncation=True)["input_ids"]

    inputs = tokenizer(list(texts), return_tensors='pt', padding='max_length', max_length=max_length, truncation=True)
    
    #With GPT tokenizer we need to manually add <bos> and <eos> tokens
    if config['is_causal_lm']:
        for i in range(len(labels)):
            temp = []
            j = 1
            while inputs['input_ids'][i][j+1] != other_token and inputs['input_ids'][i][j+1] != you_token:
                temp += [-100]
                j += 1
            labels[i] = temp + [bos_token] + labels[i]
            labels[i].extend([eos_token])
            labels[i].extend([-100 for _ in range(len(labels[i]), max_length)])

    labels = torch.LongTensor(labels)

    if 'token_type_ids' in inputs:
        inputs.pop('token_type_ids')

    inputs['labels'] = labels

    return inputs


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.texts = list(data['context'])
        self.labels = list(data['labels'])
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        if hasattr(self, 'labels'):
            return self.texts[idx], self.labels[idx]
        else:
            return self.texts[idx], []