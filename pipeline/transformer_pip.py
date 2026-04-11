from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score

from dataloader.data_loader import build_vocab_and_loaders
from model.MultiTaskTransformer import MultiTaskTransformer
from utils.earlystop import EarlyStopping
from utils.plot_utils import plot_confusion_matrix


def transformer_pipeline(
    ds_dict,
    n_epochs,
    batch_size=64,
    min_freq=2,
    embedding_dim=256,
    nhead=8,
    num_encoder_layers=2,
    dim_feedforward=512,
    dropout=0.3,
    max_seq_len=512,
    lr=1e-3,
    weight_decay=0.0,
    sent_loss_weight=1.0,
    topic_loss_weight=1.0,
    grad_clip_norm=1.0,
    scheduler_factor=0.5,
    scheduler_patience=2,
    scheduler_min_lr=1e-5,
    early_stopping_patience=10,
    early_stopping_verbose=True,
    early_stopping_delta=0.0,
    checkpoint_path="./outputs/transformer_multitask_best_model.pt",
    sentiment_names=None,
    topic_names=None,
    sentiment_report_path="./logs/transformer_sentiment_classification_report.txt",
    topic_report_path="./logs/transformer_topic_classification_report.txt",
    confusion_matrix_dir="./images/transformer",
):
    vocab, loaders = build_vocab_and_loaders(
        ds_dict=ds_dict,
        min_freq=min_freq,
        batch_size=batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if sentiment_names is None:
        sentiment_names = ["negative", "positive"]
    if topic_names is None:
        topic_names = ["lecturer", "training_program", "facility"]

    n_sentiment = len(sentiment_names)
    n_topic = len(topic_names)

    model = MultiTaskTransformer(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        num_sentiment_labels=n_sentiment,
        num_topic_labels=n_topic,
        dropout=dropout,
        max_seq_len=max_seq_len,
    ).to(device)

    criterion_sent = nn.CrossEntropyLoss()
    criterion_topic = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr,
    )

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(sentiment_report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(topic_report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(confusion_matrix_dir).mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStopping(
        save_path=checkpoint_path,
        patience=early_stopping_patience,
        verbose=early_stopping_verbose,
        delta=early_stopping_delta,
    )
    best_score = -1.0

    print(f"STARTING TRANSFORMER TRAINING ON {device}...")

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        train_correct_sent, train_correct_topic = 0, 0
        train_total = 0

        for batch in loaders["train"]:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_sent = batch["sentiment"].to(device)
            labels_topic = batch["topic"].to(device)

            optimizer.zero_grad()

            out_sent, out_topic = model(input_ids, attention_mask)
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
                attention_mask = batch["attention_mask"].to(device)
                labels_sent = batch["sentiment"].to(device)
                labels_topic = batch["topic"].to(device)

                out_sent, out_topic = model(input_ids, attention_mask)

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

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if val_score > best_score:
            best_score = val_score

        print(f"Epoch: {epoch + 1:02}")
        print(
            f"\tTrain Loss: {avg_train_loss:.4f} | "
            f"Sent Acc: {train_acc_sent * 100:.2f}% | "
            f"Topic Acc: {train_acc_topic * 100:.2f}%"
        )
        print(
            f"\t Val. Loss: {avg_val_loss:.4f} | "
            f"Sent Acc: {val_acc_sent * 100:.2f}% | "
            f"Topic Acc: {val_acc_topic * 100:.2f}%"
        )
        print(
            f"\tF1 Macro Sent/Top: {val_f1_sent:.4f} / {val_f1_topic:.4f} | "
            f"F1 Combined: {val_score:.4f} | LR: {current_lr:.2e}"
        )

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(
                f">>> Early stopping triggered at epoch {epoch + 1} "
                f"(best F1 Combined observed: {best_score:.4f})"
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
        title="transformer_sentiment",
        output_dir=confusion_matrix_dir,
    )
    plot_confusion_matrix(
        y_topic_true,
        y_topic_pred,
        classes=topic_names,
        title="transformer_topic",
        output_dir=confusion_matrix_dir,
    )

    print(">>> Transformer evaluation complete. Reports and confusion matrices saved.")

    return model


def predict(model, loader, device):
    model.eval()

    sent_preds, sent_true = [], []
    topic_preds, topic_true = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            p_sent, p_topic = model(input_ids, attention_mask)

            sent_preds.extend(p_sent.argmax(dim=1).cpu().numpy())
            sent_true.extend(batch["sentiment"].numpy())

            topic_preds.extend(p_topic.argmax(dim=1).cpu().numpy())
            topic_true.extend(batch["topic"].numpy())

    return (sent_true, sent_preds), (topic_true, topic_preds)
