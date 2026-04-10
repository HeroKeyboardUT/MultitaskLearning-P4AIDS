import torch


def epoch_train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc_sent = 0
    epoch_acc_topic = 0
    
    model.train()
    
    for batch in iterator:
        # 1. Xóa gradient cũ
        optimizer.zero_grad()
        
        # 2. Đưa dữ liệu lên Device (CPU/GPU)
        text = batch['input_ids'].to(device)
        lengths = batch['lengths'].to(device)
        y_sent = batch['sentiment'].to(device)
        y_topic = batch['topic'].to(device)
        
        # 3. Forward pass
        # Model trả về 2 logits riêng biệt
        preds_sent, preds_topic = model(text, lengths)
        
        # 4. Tính loss cho cả 2 nhiệm vụ
        loss = criterion(preds_sent, preds_topic, y_sent, y_topic)
        
        # 5. Backward & Optimizer
        loss.backward()
        optimizer.step()
        
        # 6. Lưu chỉ số để theo dõi
        acc_sent = categorical_accuracy(preds_sent, y_sent)
        acc_topic = categorical_accuracy(preds_topic, y_topic)
        
        epoch_loss += loss.item()
        epoch_acc_sent += acc_sent.item()
        epoch_acc_topic += acc_topic.item()
        
    return epoch_loss / len(iterator), epoch_acc_sent / len(iterator), epoch_acc_topic / len(iterator)

def epoch_evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc_sent = 0
    epoch_acc_topic = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            text = batch['input_ids'].to(device)
            lengths = batch['lengths'].to(device)
            y_sent = batch['sentiment'].to(device)
            y_topic = batch['topic'].to(device)
            
            preds_sent, preds_topic = model(text, lengths)
            
            loss = criterion(preds_sent, preds_topic, y_sent, y_topic)
            
            epoch_loss += loss.item()
            epoch_acc_sent += categorical_accuracy(preds_sent, y_sent).item()
            epoch_acc_topic += categorical_accuracy(preds_topic, y_topic).item()
            
    return epoch_loss / len(iterator), epoch_acc_sent / len(iterator), epoch_acc_topic / len(iterator)

def categorical_accuracy(preds, y):
    max_preds = preds.argmax(dim=1, keepdim=True)
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]]).to(preds.device)

def get_criterion(criterion_sent, criterion_topic):
    '''Returns a combined loss function for both sentiment and topic classification.
    Args:
        criterion_sent: Loss function for sentiment classification (e.g., nn.CrossEntropyLoss()).
        criterion_topic: Loss function for topic classification (e.g., nn.CrossEntropyLoss()).
    Returns:
        A function that takes predictions and true labels for both tasks and returns the total loss.
            args:
                preds_sent: Logits from the sentiment classification head (shape: [batch_size, num_sentiments])
                preds_topic: Logits from the topic classification head (shape: [batch_size, num_topics])
                y_sent: True labels for sentiment classification (shape: [batch_size])
                y_topic: True labels for topic classification (shape: [batch_size])'''
    return lambda preds_sent, preds_topic, y_sent, y_topic: criterion_sent(preds_sent, y_sent) + criterion_topic(preds_topic, y_topic)

