from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def plot_confusion_matrix(y_true, y_pred, classes, title, output_dir="images"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix: {title}')
    plt.ylabel('Ground Truth (True)')
    plt.xlabel('Predicted (Pred)')
    # Lưu confusion matrix vào file
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe_title = title.lower().replace(" ", "_")
    plt.savefig(f'{output_dir}/{safe_title}_confusion_matrix.png')
    plt.close()