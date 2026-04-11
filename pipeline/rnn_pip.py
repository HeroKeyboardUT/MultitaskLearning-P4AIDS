import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import classification_report, f1_score

from dataloader.data_loader import build_vocab_and_loaders
from model.BidirRNN import BidirRNN
from utils.plot_utils import plot_confusion_matrix


def rnn_pipline(
    ds_dict,
    n_epochs,
    batch_size=64,
    min_freq=2,
    embedding_dim=128,
    hidden_dim=96,
    n_layers=2,
    dropout=0.35,
    bidirectional=True,
    rnn_type="gru",
    lr=5e-4,
    weight_decay=1e-4,
    sent_loss_weight=1.0,
    topic_loss_weight=1.5,
    grad_clip_norm=1.0,
    scheduler_factor=0.5,
    scheduler_patience=2,
    scheduler_min_lr=1e-5,
    early_stopping_patience=5,
    checkpoint_path="./outputs/best_multitask_rnn.pt",
    sentiment_names=None,
    topic_names=None,
    sentiment_report_path="./logs/rnn_sentiment_report.txt",
    topic_report_path="./logs/rnn_topic_report.txt",
    confusion_matrix_dir="./images",
):
    # Xay dung vocab va dataloader
    vocab, loaders = build_vocab_and_loaders(
        ds_dict=ds_dict,
        batch_size=batch_size,
        min_freq=min_freq,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if sentiment_names is None:
        sentiment_names = ["negative", "positive"]
    if topic_names is None:
        topic_names = ["lecturer", "training_program", "facility"]

    n_sentiment = len(sentiment_names)
    n_topic = len(topic_names)

    train_sent = torch.tensor(ds_dict["train"]["sentiment"], dtype=torch.long)
    train_topic = torch.tensor(ds_dict["train"]["topic"], dtype=torch.long)

    sent_counts = torch.bincount(train_sent, minlength=n_sentiment).float()
    topic_counts = torch.bincount(train_topic, minlength=n_topic).float()

    weight_sent = (sent_counts.sum() / (n_sentiment * sent_counts.clamp_min(1.0))).to(device)
    weight_topic = (topic_counts.sum() / (n_topic * topic_counts.clamp_min(1.0))).to(device)

    model = BidirRNN(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        num_sentiment_labels=n_sentiment,
        num_topic_labels=n_topic,
        dropout=dropout,
        bidirectional=bidirectional,
        rnn_type=rnn_type,
    ).to(device)

    criterion_sent = nn.CrossEntropyLoss(weight=weight_sent)
    criterion_topic = nn.CrossEntropyLoss(weight=weight_topic)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr,
    )

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(sentiment_report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(topic_report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(confusion_matrix_dir).mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    bad_epochs = 0

    print(f"STARTING RNN TRAINING ON {device}...")
    print(f"Using {rnn_type.upper()} backbone")

    for epoch in range(n_epochs):
        model.train()

        train_loss = 0.0
        train_correct_sent, train_correct_topic = 0, 0
        train_total = 0

        for batch in loaders["train"]:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"]
            labels_sent = batch["sentiment"].to(device)
            labels_topic = batch["topic"].to(device)

            optimizer.zero_grad()

            out_sent, out_topic = model(input_ids, lengths)
            loss_sent = criterion_sent(out_sent, labels_sent)
            loss_topic = criterion_topic(out_topic, labels_topic)
            total_loss = sent_loss_weight * loss_sent + topic_loss_weight * loss_topic

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            train_loss += total_loss.item()
            train_correct_sent += (out_sent.argmax(dim=1) == labels_sent).sum().item()
            train_correct_topic += (out_topic.argmax(dim=1) == labels_topic).sum().item()
            train_total += labels_sent.size(0)

        avg_train_loss = train_loss / len(loaders["train"])
        train_acc_sent = train_correct_sent / train_total
        train_acc_topic = train_correct_topic / train_total

        model.eval()
        val_loss = 0.0
        val_correct_sent, val_correct_topic = 0, 0
        val_total = 0

        y_sent_true, y_sent_pred = [], []
        y_topic_true, y_topic_pred = [], []

        with torch.no_grad():
            for batch in loaders["validation"]:
                input_ids = batch["input_ids"].to(device)
                lengths = batch["lengths"]
                labels_sent = batch["sentiment"].to(device)
                labels_topic = batch["topic"].to(device)

                out_sent, out_topic = model(input_ids, lengths)

                loss_sent = criterion_sent(out_sent, labels_sent)
                loss_topic = criterion_topic(out_topic, labels_topic)
                total_loss = sent_loss_weight * loss_sent + topic_loss_weight * loss_topic

                val_loss += total_loss.item()

                preds_sent = out_sent.argmax(dim=1)
                preds_topic = out_topic.argmax(dim=1)

                val_correct_sent += (preds_sent == labels_sent).sum().item()
                val_correct_topic += (preds_topic == labels_topic).sum().item()
                val_total += labels_sent.size(0)

                y_sent_true.extend(labels_sent.cpu().tolist())
                y_sent_pred.extend(preds_sent.cpu().tolist())
                y_topic_true.extend(labels_topic.cpu().tolist())
                y_topic_pred.extend(preds_topic.cpu().tolist())

        avg_val_loss = val_loss / len(loaders["validation"])
        val_acc_sent = val_correct_sent / val_total
        val_acc_topic = val_correct_topic / val_total

        val_f1_sent = f1_score(y_sent_true, y_sent_pred, average="macro", zero_division=0)
        val_f1_topic = f1_score(y_topic_true, y_topic_pred, average="macro", zero_division=0)
        val_score = (val_f1_sent + val_f1_topic) / 2.0

        scheduler.step(val_score)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch: {epoch + 1:02} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} || "
            f"Sent Acc (Train/Val): {train_acc_sent * 100:.2f}% / {val_acc_sent * 100:.2f}% | "
            f"Topic Acc (Train/Val): {train_acc_topic * 100:.2f}% / {val_acc_topic * 100:.2f}% || "
            f"F1 Sent/Topic: {val_f1_sent:.4f} / {val_f1_topic:.4f} | "
            f"F1 Combined: {val_score:.4f} | LR: {current_lr:.2e}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), checkpoint_path)
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= early_stopping_patience:
            print(
                f">>> Early stopping at epoch {epoch + 1} "
                f"(best F1 Combined: {best_score:.4f})"
            )
            break

    print(
        f">>> Loading best model (F1 Combined={best_score:.4f}) for test evaluation..."
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print(">>> Evaluating best model on test set...")
    (y_sent_true, y_sent_pred), (y_topic_true, y_topic_pred) = predict(model, loaders["test"], device)

    with open(sentiment_report_path, "w", encoding="utf-8") as f:
        f.write(
            classification_report(
                y_sent_true,
                y_sent_pred,
                target_names=sentiment_names,
                zero_division=0,
            )
        )

    with open(topic_report_path, "w", encoding="utf-8") as f:
        f.write(
            classification_report(
                y_topic_true,
                y_topic_pred,
                target_names=topic_names,
                zero_division=0,
            )
        )

    plot_confusion_matrix(
        y_sent_true,
        y_sent_pred,
        classes=sentiment_names,
        title="RNN Sentiment Confusion Matrix",
        output_dir=confusion_matrix_dir,
    )
    plot_confusion_matrix(
        y_topic_true,
        y_topic_pred,
        classes=topic_names,
        title="RNN Topic Confusion Matrix",
        output_dir=confusion_matrix_dir,
    )

    print(">>> RNN evaluation complete. Reports and confusion matrices saved.")

    return model


def predict(model, loader, device):
    model.eval()

    sent_preds, sent_true = [], []
    topic_preds, topic_true = [], []

    with torch.no_grad():
        for batch in loader:
            text = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)

            p_sent, p_topic = model(text, lengths)

            sent_preds.extend(p_sent.argmax(dim=1).cpu().numpy())
            sent_true.extend(batch["sentiment"].numpy())

            topic_preds.extend(p_topic.argmax(dim=1).cpu().numpy())
            topic_true.extend(batch["topic"].numpy())

    return (sent_true, sent_preds), (topic_true, topic_preds)
