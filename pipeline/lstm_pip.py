import torch
import torch.nn as nn
from pathlib import Path
from dataloader.data_loader import build_vocab_and_loaders
from utils.earlystop import EarlyStopping
from model.BidirLSTM import BidirLSTM
from utils.model_ultis import epoch_train, epoch_evaluate, get_criterion
from sklearn.metrics import classification_report
from utils.plot_utils import plot_confusion_matrix


def lstm_pipline(
    ds_dict,
    n_epochs,
    batch_size=64,
    min_freq=2,
    sent_alpha=0.2,
    topic_alpha=0.2,
    embedding_dim=300,
    hidden_dim=256,
    n_layers=2,
    dropout=0.5,
    bidirectional=True,
    lr=1e-3,
    weight_decay=1e-3,
    early_stopping_patience=10,
    early_stopping_verbose=True,
    early_stopping_delta=0.0,
    checkpoint_path="./outputs/lstm_multitask_best_model.pt",
    sentiment_names=None,
    topic_names=None,
    sentiment_report_path="./logs/lstm_sentiment_classification_report.txt",
    topic_report_path="./logs/lstm_topic_classification_report.txt",
    confusion_matrix_dir="./images/lstm",
    use_wandb=False,
    wandb_project="student-feedback",
    wandb_run_name=None,
    wandb_tags=None,
):
    # Xây dựng vocab và dataloader
    vocab, loaders = build_vocab_and_loaders(ds_dict=ds_dict, batch_size=batch_size, min_freq=min_freq)
    # Khởi tạo Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if sentiment_names is None:
        sentiment_names = ["negative", "positive"]
    if topic_names is None:
        topic_names = ["lecturer", "training_program", "facility"]

    model = BidirLSTM(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        num_sentiment_labels=len(sentiment_names),
        num_topic_labels=len(topic_names),
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Khởi tạo loss với label smoothing
    criterion_sent = nn.CrossEntropyLoss(label_smoothing=sent_alpha)
    criterion_topic = nn.CrossEntropyLoss(label_smoothing=topic_alpha)
    criterion = get_criterion(criterion_sent, criterion_topic)

    # Khởi tạo Early Stopping
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        verbose=early_stopping_verbose,
        delta=early_stopping_delta,
        save_path=checkpoint_path,
    )

    # Fit model
    print(f"STARTING TRAINING ON {device}...")

    wandb_config_payload = {
        "model": "LSTM",
        "vocab_size": len(vocab),
        "device": str(device),
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "min_freq": min_freq,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "sentiment_label_smoothing": sent_alpha,
        "topic_label_smoothing": topic_alpha,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_delta": early_stopping_delta,
    }

    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                tags=wandb_tags,
                config=wandb_config_payload,
            )
        except ImportError:
            print("[WARN] wandb is not installed. Continuing without W&B logging.")
            use_wandb = False

    for epoch in range(n_epochs):
        train_loss, train_acc_sent, train_acc_topic = epoch_train(model, loaders["train"], optimizer, criterion, device)
        valid_loss, valid_acc_sent, valid_acc_topic = epoch_evaluate(model, loaders["validation"], criterion, device)


        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Sent Acc: {train_acc_sent*100:.2f}% | Topic Acc: {train_acc_topic*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} | Sent Acc: {valid_acc_sent*100:.2f}% | Topic Acc: {valid_acc_topic*100:.2f}%')

        if use_wandb and wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/acc_sentiment": train_acc_sent,
                "train/acc_topic": train_acc_topic,
                "val/loss": valid_loss,
                "val/acc_sentiment": valid_acc_sent,
                "val/acc_topic": valid_acc_topic,
            })

        # GỌI EARLY STOPPING
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print(">>> Early stopping triggered.")
            break
    
    # Evaluate model
    # Lấy model tốt nhất đã lưu
    print('>>> Loading best model for evaluation...')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print('>>> Evaluating best model on test set...')
    (y_sent_true, y_sent_pred), (y_topic_true, y_topic_pred) = predict(model, loaders["test"], device)

    Path(sentiment_report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(topic_report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(confusion_matrix_dir).mkdir(parents=True, exist_ok=True)

    # Lưu báo cáo phân loại cho từng task vào file txt
    with open(sentiment_report_path, "w", encoding="utf-8") as f:
        f.write(classification_report(y_sent_true, y_sent_pred, target_names=sentiment_names))
    with open(topic_report_path, "w", encoding="utf-8") as f:
        f.write(classification_report(y_topic_true, y_topic_pred, target_names=topic_names))

    # Vẽ confusion matrix cho từng task
    plot_confusion_matrix(
        y_sent_true,
        y_sent_pred,
        classes=sentiment_names,
        title="lstm_sentiment",
        output_dir=confusion_matrix_dir,
    )
    plot_confusion_matrix(
        y_topic_true,
        y_topic_pred,
        classes=topic_names,
        title="lstm_topic",
        output_dir=confusion_matrix_dir,
    )
    
    print('>>> Evaluation complete. Reports and confusion matrices saved.')

    if use_wandb and wandb_run is not None:
        wandb_run.log({"best/val_loss": early_stopping.val_loss_min})
        wandb_run.finish()

    return model

def predict(model, loader, device):
    model.eval()
    
    # List lưu kết quả cho từng task
    sent_preds, sent_true = [], []
    topic_preds, topic_true = [], []
    
    with torch.no_grad():
        for batch in loader:
            text = batch['input_ids'].to(device)
            lengths = batch['lengths'].to(device)
            
            # Predict
            p_sent, p_topic = model(text, lengths)
            
            # Lấy index của class có xác suất cao nhất
            sent_preds.extend(p_sent.argmax(dim=1).cpu().numpy())
            sent_true.extend(batch['sentiment'].numpy())
            
            topic_preds.extend(p_topic.argmax(dim=1).cpu().numpy())
            topic_true.extend(batch['topic'].numpy())
            
    return (sent_true, sent_preds), (topic_true, topic_preds)