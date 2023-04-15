import torch
from tqdm import tqdm

def train(model,train_dataloader,optimizer,writer,epoch,max_grad_norm=None):
  loss_fn = torch.nn.CrossEntropyLoss()
  num_batches = len(train_dataloader)
  log_interval = 1
  
  model.train()
  with torch.enable_grad():
    for idx,batch in enumerate(tqdm(train_dataloader,position=1)):
      # Zero the gradients and clear the accumulated loss
      optimizer.zero_grad()

      # Move to device
      batch = tuple(t.to(device) for t in batch)
      query_input_ids,query_attention_mask,query_label,support_input_ids,support_attention_mask,support_label = batch

      # Compute loss
      pred = model(query_input_ids,query_attention_mask, support_input_ids,support_attention_mask,support_label)
      loss = loss_fn(pred, query_label)
      loss.backward()


      # Clip gradients if necessary
      if max_grad_norm is not None:
          clip_grad_norm_(model.parameters(), max_grad_norm)

      # Optimize
      optimizer.step()

      
      # Log training loss
      train_loss = loss.item()
      if log_interval > 0:
        if idx % log_interval == 0:
            global_step = idx + (epoch * num_batches)
            writer.add_scalar('Training/Loss_IT', train_loss, global_step)

    # Zero the gradients when exiting a train step
    optimizer.zero_grad()
  return loss.item()