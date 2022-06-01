import json
import csv
import math
import torch
import transformers
import numpy as np
import torch.nn.functional as F
from util import count_parameters, compute_metrics, compute_metrics_from_logits
# title match agregate fuse
def cprint(*args):
    text = ""
    for arg in args:
        text += "{0} ".format(arg)
    print(text)
    
def match(model, matching_method, x, y, x_mask, y_mask):
    # Multi-hop Co-Attention
    # x: (batch_size, m, hidden_size)
    # y: (batch_size, n, hidden_size)
    # x_mask: (batch_size, m)
    # y_mask: (batch_size, n)
    assert x.dim() == 3 and y.dim() == 3
    assert x_mask.dim() == 2 and y_mask.dim() == 2
    assert x_mask.shape == x.shape[:2] and y_mask.shape == y.shape[:2]
    m = x.shape[1]
    n = y.shape[1]
    attn_mask = torch.bmm(x_mask.unsqueeze(-1), y_mask.unsqueeze(1)) # (batch_size, m, n)
    attn = torch.bmm(x, y.transpose(1,2)) # (batch_size, m, n)
    model.attn = attn
    model.attn_mask = attn_mask
    
    x_to_y = torch.softmax(attn * attn_mask + (-5e4) * (1-attn_mask), dim=2) # (batch_size, m, n)
    y_to_x = torch.softmax(attn * attn_mask + (-5e4) * (1-attn_mask), dim=1).transpose(1,2) # # (batch_size, n, m)
    
    # x_attended, y_attended = None, None # no hop-1
    x_attended = torch.bmm(x_to_y, y) # (batch_size, m, hidden_size)
    y_attended = torch.bmm(y_to_x, x) # (batch_size, n, hidden_size)
    # x_attended_2hop, y_attended_2hop = None, None # no hop-2
    y_attn = torch.bmm(y_to_x.mean(dim=1, keepdim=True), x_to_y) # (batch_size, 1, n) # true important attention over y
    x_attn = torch.bmm(x_to_y.mean(dim=1, keepdim=True), y_to_x) # (batch_size, 1, m) # true important attention over x
    # truly attended representation
    x_attended_2hop = torch.bmm(x_attn, x).squeeze(1) # (batch_size, hidden_size)
    y_attended_2hop = torch.bmm(y_attn, y).squeeze(1) # (batch_size, hidden_size)
    # # hop-3
    # y_attn, x_attn = torch.bmm(x_attn, x_to_y), torch.bmm(y_attn, y_to_x) # (batch_size, 1, n) # true important attention over y
    # x_attended_3hop = torch.bmm(x_attn, x).squeeze(1) # (batch_size, hidden_size)
    # y_attended_3hop = torch.bmm(y_attn, y).squeeze(1) # (batch_size, hidden_size)
    # x_attended_2hop = torch.cat([x_attended_2hop, x_attended_3hop], dim=-1)
    # y_attended_2hop = torch.cat([y_attended_2hop, y_attended_3hop], dim=-1)
    x_attended = x_attended, x_attended_2hop
    y_attended = y_attended, y_attended_2hop
    return x_attended, y_attended
def aggregate(model, aggregation_method, x, x_mask):
    # x: (batch_size, seq_len, emb_size)
    # x_mask: (batch_size, seq_len)
    assert x.dim() == 3 and x_mask.dim() == 2
    assert x.shape[:2] == x_mask.shape
    # batch_size, seq_len, emb_size = x.shape
    if aggregation_method == "mean":
        return (x * x_mask.unsqueeze(-1)).sum(dim=1)/x_mask.sum(dim=-1, keepdim=True).clamp(min=1) # (batch_size, emb_size)
    if aggregation_method == "max":
        return x.masked_fill(x_mask.unsqueeze(-1)==0, -5e4).max(dim=1)[0] # (batch_size, emb_size)
    if aggregation_method == "mean_max":
        return torch.cat([(x * x_mask.unsqueeze(-1)).sum(dim=1)/x_mask.sum(dim=-1, keepdim=True).clamp(min=1), \
            x.masked_fill(x_mask.unsqueeze(-1)==0, -5e4).max(dim=1)[0]], dim=-1) # (batch_size, 2*emb_size)
    if aggregation_method == "cls":
        return x[:,0] # (batch_size, emb_size)
    if aggregation_method == "cls_gpt":
        return x[:,-1] # (batch_size, emb_size)
def fuse(model, matching_method, aggregation_method, batch_x_emb, batch_y_emb, batch_persona_emb, \
    batch_x_mask, batch_y_mask, batch_persona_mask, batch_size, num_candidates):
    
    batch_x_emb, batch_y_emb_context = match(model, matching_method, batch_x_emb, batch_y_emb, batch_x_mask, batch_y_mask)
    # batch_x_emb: ((batch_size*num_candidates, m, emb_size), (batch_size*num_candidates, emb_size))
    # batch_y_emb_context: (batch_size*num_candidates, n, emb_size), (batch_size*num_candidates, emb_size)
    
    # hop 2 results
    batch_x_emb_2hop = batch_x_emb[1]
    batch_y_emb_context_2hop = batch_y_emb_context[1]
    
    # mean_max aggregation for the 1st hop result
    batch_x_emb = aggregate(model, aggregation_method, batch_x_emb[0], batch_x_mask) # batch_x_emb: (batch_size*num_candidates, 2*emb_size)
    batch_y_emb_context = aggregate(model, aggregation_method, batch_y_emb_context[0], batch_y_mask) # batch_y_emb_context: (batch_size*num_candidates, 2*emb_size)
    if batch_persona_emb is not None:
        batch_persona_emb, batch_y_emb_persona = match(model, matching_method, batch_persona_emb, batch_y_emb, batch_persona_mask, batch_y_mask)
        # batch_persona_emb: (batch_size*num_candidates, m, emb_size), (batch_size*num_candidates, emb_size)
        # batch_y_emb_persona: (batch_size*num_candidates, n, emb_size), (batch_size*num_candidates, emb_size)
        batch_persona_emb_2hop = batch_persona_emb[1]
        batch_y_emb_persona_2hop = batch_y_emb_persona[1]
        # # no hop-1
        # return torch.bmm(torch.cat([batch_x_emb_2hop, batch_persona_emb_2hop], dim=-1).unsqueeze(1), \
        #             torch.cat([batch_y_emb_context_2hop, batch_y_emb_persona_2hop], dim=-1)\
        #                 .unsqueeze(-1)).reshape(batch_size, num_candidates)
        
        batch_persona_emb = aggregate(model, aggregation_method, batch_persona_emb[0], batch_persona_mask) # batch_persona_emb: (batch_size*num_candidates, 2*emb_size)
        batch_y_emb_persona = aggregate(model, aggregation_method, batch_y_emb_persona[0], batch_y_mask) # batch_y_emb_persona: (batch_size*num_candidates, 2*emb_size)
        # # no hop-2
        # return torch.bmm(torch.cat([batch_x_emb, batch_persona_emb], dim=-1).unsqueeze(1), \
        #             torch.cat([batch_y_emb_context, batch_y_emb_persona], dim=-1)\
        #                 .unsqueeze(-1)).reshape(batch_size, num_candidates)
        return torch.bmm(torch.cat([batch_x_emb, batch_x_emb_2hop, batch_persona_emb, batch_persona_emb_2hop], dim=-1).unsqueeze(1), \
                    torch.cat([batch_y_emb_context, batch_y_emb_context_2hop, batch_y_emb_persona, batch_y_emb_persona_2hop], dim=-1)\
                        .unsqueeze(-1)).reshape(batch_size, num_candidates)
    else:
        return torch.bmm(torch.cat([batch_x_emb, batch_x_emb_2hop], dim=-1).unsqueeze(1), \
                    torch.cat([batch_y_emb_context, batch_y_emb_context_2hop], dim=-1)\
                        .unsqueeze(-1)).reshape(batch_size, num_candidates)
def dot_product_loss(batch_x_emb, batch_y_emb):
    """
        if batch_x_emb.dim() == 2:
            # batch_x_emb: (batch_size, emb_size)
            # batch_y_emb: (batch_size, emb_size)
        
        if batch_x_emb.dim() == 3:
            # batch_x_emb: (batch_size, batch_size, emb_size), the 1st dim is along examples and the 2nd dim is along candidates
            # batch_y_emb: (batch_size, emb_size)
    """
    batch_size = batch_x_emb.size(0)
    targets = torch.arange(batch_size, device=batch_x_emb.device)
    if batch_x_emb.dim() == 2:
        dot_products = batch_x_emb.mm(batch_y_emb.t())
    elif batch_x_emb.dim() == 3:
        dot_products = torch.bmm(batch_x_emb, batch_y_emb.unsqueeze(0).repeat(batch_size, 1, 1).transpose(1,2))[:, targets, targets] # (batch_size, batch_size)
    
    # dot_products: [batch, batch]
    log_prob = F.log_softmax(dot_products, dim=1)
    loss = F.nll_loss(log_prob, targets)
    nb_ok = (log_prob.max(dim=1)[1] == targets).float().sum()
    return loss, nb_ok
# title train
def train_epoch(data_iter, models, has_persona, optimizers, schedulers, gradient_accumulation_steps, device, fp16, amp, \
    apply_interaction, matching_method, aggregation_method):
    models = [i.to(device) for i in models]
    epoch_loss = []
    ok = 0
    total = 0
    print_every = 1000
    if len(models) == 1:
        if has_persona == 0:
            context_model, response_model = models[0], models[0]
        else:
            context_model, response_model, persona_model = models[0], models[0], models[0]
    if len(models) == 2:
        context_model, response_model = models
    if len(models) == 3:
        context_model, response_model, persona_model = models
    
    for optimizer in optimizers:
        optimizer.zero_grad()
    for i, batch in enumerate(data_iter):
        batch, labels = batch
        batch_x = {k:batch['context'][k].to(device) for k in batch['context']}
        batch_y = {k:batch['responce'][k].to(device) for k in batch['responce']}
        if has_persona:
            batch_persona = {k:batch['persona'][k].to(device) for k in batch['persona']}
        
        # batch_x = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
        # batch_y = {"input_ids": batch[3], "attention_mask": batch[4], "token_type_ids": batch[5]}
        # batch_persona = {"input_ids": batch[6], "attention_mask": batch[7], "token_type_ids": batch[8]}
        #if i == 0:
         #   print(batch_x['input_ids'][0])
        #print([k+'='+str(batch_x[k].size()) for k in batch_x]) 
        output_x = context_model(**batch_x)
        output_y = response_model(**batch_y)
        
        if apply_interaction:
            # batch_x_mask = batch[0].ne(0).float()
            # batch_y_mask = batch[3].ne(0).float()
            batch_x_mask = batch_x['attention_mask'].float()
            batch_y_mask = batch_y['attention_mask'].float()
            batch_x_emb = output_x[0] # (batch_size, context_len, emb_size)
            batch_y_emb = output_y[0] # (batch_size, sent_len, emb_size)
            batch_size, sent_len, emb_size = batch_y_emb.shape
            batch_persona_emb = None
            batch_persona_mask = None
            num_candidates = batch_size
            if has_persona:
                # batch_persona_mask = batch[6].ne(0).float()
                batch_persona_mask = batch_persona['attention_mask'].float()
                output_persona = persona_model(**batch_persona)
                batch_persona_emb = output_persona[0] # (batch_size, persona_len, emb_size)
                batch_persona_emb = batch_persona_emb.repeat_interleave(num_candidates, dim=0)
                batch_persona_mask = batch_persona_mask.repeat_interleave(num_candidates, dim=0)
            batch_x_emb = batch_x_emb.repeat_interleave(num_candidates, dim=0) # (batch_size*num_candidates, context_len, emb_size)
            batch_x_mask = batch_x_mask.repeat_interleave(num_candidates, dim=0) # (batch_size*num_candidates, context_len)
            
            # interaction
            # context-response attention
            batch_y_emb = batch_y_emb.unsqueeze(0).repeat(batch_size, 1, 1, 1).reshape(-1, sent_len, emb_size) # (batch_size*num_candidates, sent_len, emb_size)
            batch_y_mask = batch_y_mask.unsqueeze(0).repeat(batch_size, 1, 1).reshape(-1, sent_len) # (batch_size*num_candidates, sent_len)
            logits = fuse(context_model, matching_method, aggregation_method, \
                batch_x_emb, batch_y_emb, batch_persona_emb, batch_x_mask, batch_y_mask, batch_persona_mask, batch_size, num_candidates)
            
            # compute loss
            targets = torch.arange(batch_size, dtype=torch.long, device=batch_x['input_ids'].device)
            loss = F.cross_entropy(logits, targets)
            num_ok = (targets.long() == logits.float().argmax(dim=1)).sum()
        else:
            batch_x_emb = output_x[0].mean(dim=1) # batch_x_emb: (batch_size, emb_size)
            batch_y_emb = output_y[0].mean(dim=1)
            if has_persona:
                output_persona = persona_model(**batch_persona)
                batch_persona_emb = output_persona[0].mean(dim=1)
                batch_x_emb = (batch_x_emb + batch_persona_emb)/2
            
            # compute loss
            loss, num_ok = dot_product_loss(batch_x_emb, batch_y_emb)
        
        ok += num_ok.item()
        total += batch_x['input_ids'].shape[0]
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        if fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        if (i+1) % gradient_accumulation_steps == 0:
            for model, optimizer, scheduler in zip(models, optimizers, schedulers):
                if fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                scheduler.step()
                
                # clear grads here
                for optimizer in optimizers:
                    optimizer.zero_grad()
        epoch_loss.append(loss.item())
        if i and i%print_every == 0:
            cprint("loss: ", np.mean(epoch_loss[-print_every:]))
            cprint("accuracy: ", ok/total)
    acc = ok/total
    return np.mean(epoch_loss), (acc, 0, 0)
# eval
def evaluate_epoch(data_iter, models, has_persona, gradient_accumulation_steps, device, epoch, \
    apply_interaction, matching_method, aggregation_method):
    models = [i.to(device) for i in models]
    epoch_loss = []
    ok = 0
    total = 0
    recall = []
    MRR = []
    print_every = 1000
    if len(models) == 1:
        if has_persona == 0:
            context_model, response_model = models[0], models[0]
        else:
            context_model, response_model, persona_model = models[0], models[0], models[0]
    if len(models) == 2:
        context_model, response_model = models
    if len(models) == 3:
        context_model, response_model, persona_model = models
    
    for batch_idx, batch in enumerate(data_iter):
        batch, labels = batch
        batch_x = {k:batch['context'][k].to(device) for k in batch['context']}
        batch_y = {k:batch['responce'][k].to(device) for k in batch['responce']}
        if has_persona:
            batch_persona = {k:batch['persona'][k].to(device) for k in batch['persona']}
        
        # get context embeddings in chunks due to memory constraint
        batch_size = batch_x['input_ids'].shape[0]
        chunk_size = 20
        num_chunks = math.ceil(batch_size/chunk_size)
        if apply_interaction:
            # batch_x_mask = batch[0].ne(0).float()
            # batch_y_mask = batch[3].ne(0).float()
            batch_x_mask = batch_x['attention_mask'].float()
            batch_y_mask = batch_y['attention_mask'].float()
            
            batch_x_emb = []
            batch_x_pooled_emb = []
            with torch.no_grad():
                for i in range(num_chunks):
                    mini_batch_x = {
                        "input_ids": batch_x['input_ids'][i*chunk_size: (i+1)*chunk_size], 
                        "attention_mask": batch_x['attention_mask'][i*chunk_size: (i+1)*chunk_size]#, 
                        #"token_type_ids": batch_x['token_type_ids'][i*chunk_size: (i+1)*chunk_size]
                        }
                    mini_output_x = context_model(**mini_batch_x)
                    batch_x_emb.append(mini_output_x[0]) # [(chunk_size, seq_len, emb_size), ...]
                    batch_x_pooled_emb.append(mini_output_x[1])
                batch_x_emb = torch.cat(batch_x_emb, dim=0) # (batch_size, seq_len, emb_size)
                batch_x_pooled_emb = torch.cat(batch_x_pooled_emb, dim=0)
                emb_size = batch_x_emb.shape[-1]
            if has_persona:
                # batch_persona_mask = batch[6].ne(0).float()
                batch_persona_mask = batch_persona['attention_mask'].float()
                batch_persona_emb = []
                batch_persona_pooled_emb = []
                with torch.no_grad():
                    for i in range(num_chunks):
                        mini_batch_persona = {
                            "input_ids": batch_persona['input_ids'][i*chunk_size: (i+1)*chunk_size], 
                            "attention_mask": batch_persona['attention_mask'][i*chunk_size: (i+1)*chunk_size]#, 
                            #"token_type_ids": batch_persona['token_type_ids'][i*chunk_size: (i+1)*chunk_size]
                            }
                        mini_output_persona = persona_model(**mini_batch_persona)
                        # [(chunk_size, emb_size), ...]
                        batch_persona_emb.append(mini_output_persona[0])
                        batch_persona_pooled_emb.append(mini_output_persona[1])
                    batch_persona_emb = torch.cat(batch_persona_emb, dim=0)
                    batch_persona_pooled_emb = torch.cat(batch_persona_pooled_emb, dim=0)
            with torch.no_grad():
                output_y = response_model(**batch_y)
                batch_y_emb = output_y[0]
            batch_size, sent_len, emb_size = batch_y_emb.shape
            # interaction
            # context-response attention
            num_candidates = batch_size
            
            with torch.no_grad():
                # evaluate per example
                logits = []
                for i in range(batch_size):
                    x_emb = batch_x_emb[i:i+1].repeat_interleave(num_candidates, dim=0) # (num_candidates, context_len, emb_size)
                    x_mask = batch_x_mask[i:i+1].repeat_interleave(num_candidates, dim=0) # (batch_size*num_candidates, context_len)
                    persona_emb, persona_mask = None, None
                    if has_persona:
                        persona_emb = batch_persona_emb[i:i+1].repeat_interleave(num_candidates, dim=0)
                        persona_mask = batch_persona_mask[i:i+1].repeat_interleave(num_candidates, dim=0)
                    logits_single = fuse(context_model, matching_method, aggregation_method, \
                        x_emb, batch_y_emb, persona_emb, x_mask, batch_y_mask, persona_mask, 1, num_candidates).reshape(-1)
                    
                    logits.append(logits_single)
                logits = torch.stack(logits, dim=0)
                
                # compute loss
                targets = torch.arange(batch_size, dtype=torch.long, device=batch_x['input_ids'].device)
                loss = F.cross_entropy(logits, targets)
            num_ok = (targets.long() == logits.float().argmax(dim=1)).sum()
            valid_recall, valid_MRR = compute_metrics_from_logits(logits, targets)
        else:
            batch_x_emb = []
            with torch.no_grad():
                for i in range(num_chunks):
                    mini_batch_x = {
                        "input_ids": batch_x['input_ids'][i*chunk_size: (i+1)*chunk_size], 
                        "attention_mask": batch_x['attention_mask'][i*chunk_size: (i+1)*chunk_size]#, 
                        #"token_type_ids": batch_x['token_type_ids'][i*chunk_size: (i+1)*chunk_size]
                        }
                    mini_output_x = context_model(**mini_batch_x)
                    batch_x_emb.append(mini_output_x[0].mean(dim=1)) # [(chunk_size, emb_size), ...]
                batch_x_emb = torch.cat(batch_x_emb, dim=0) # (batch_size, emb_size)
                emb_size = batch_x_emb.shape[-1]
            if has_persona:
                batch_persona_emb = []
                with torch.no_grad():
                    for i in range(num_chunks):
                        mini_batch_persona = {
                            "input_ids": batch_persona['input_ids'][i*chunk_size: (i+1)*chunk_size], 
                            "attention_mask": batch_persona['attention_mask'][i*chunk_size: (i+1)*chunk_size]#, 
                            #"token_type_ids": batch_persona['token_type_ids'][i*chunk_size: (i+1)*chunk_size]
                            }
                        mini_output_persona = persona_model(**mini_batch_persona)
                        # [(chunk_size, emb_size), ...]
                        batch_persona_emb.append(mini_output_persona[0].mean(dim=1))
                       
            with torch.no_grad():
                batch_persona_emb = torch.cat(batch_persona_emb, dim=0)
                batch_x_emb = (batch_x_emb + batch_persona_emb)/2
                
                output_y = response_model(**batch_y)
                batch_y_emb = output_y[0].mean(dim=1)
            # compute loss
            loss, num_ok = dot_product_loss(batch_x_emb, batch_y_emb)
            valid_recall, valid_MRR = compute_metrics(batch_x_emb, batch_y_emb)
        
        ok += num_ok.item()
        total += batch_x['input_ids'].shape[0]
        # compute valid recall
        recall.append(valid_recall)
        MRR.append(valid_MRR)
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        epoch_loss.append(loss.item())
        if batch_idx and batch_idx%print_every == 0:
            cprint("loss: ", np.mean(epoch_loss[-print_every:]))
            cprint("valid recall: ", np.mean(recall[-print_every:], axis=0))
            cprint("valid MRR: ", np.mean(MRR[-print_every:], axis=0))
    acc = ok/total
    # compute recall for validation dataset
    recall = np.mean(recall, axis=0)
    MRR = np.mean(MRR)
    return np.mean(epoch_loss), (acc, recall, MRR)
