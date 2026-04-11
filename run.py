"""
Main entry point for Student Feedback pipeline.

This script reads configuration, handles CLI params, optionally downloads data,
and executes selected pipeline tasks.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

# Add utils to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))

from data_utils import download_data
from data_utils import load_splits
from data_utils import visualize_data
from pipeline.lstm_pip import lstm_pipline
from pipeline.rnn_pip import rnn_pipline


def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level))

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, "execution.log"))
    fh.setLevel(getattr(logging, log_level))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level))

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def create_directories(paths: dict) -> None:
    """
    Create all necessary directories.

    Args:
        paths: Dictionary with directory paths
    """
    for dir_name, dir_path in paths.items():
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def run_download(config: dict, logger: logging.Logger, overwrite: bool = False) -> None:
    """
    Download raw data files as a standalone step.

    Args:
        config: Configuration dictionary
        logger: Logger instance
        overwrite: If True, re-download existing files
    """
    data_config = config["data"]

    logger.info("Downloading data files...")

    download_data(
        file_ids_by_split=data_config["file_ids"],
        splits=data_config["splits"],
        data_dir=config["paths"]["data_dir"],
        file_name_patterns=data_config["file_patterns"],
        quiet=True,
        overwrite=overwrite,
    )
    logger.info("Data download completed")


def task_run_pipeline(
    config: dict,
    logger: logging.Logger,
    model_name: str,
):
    """
    Run end-to-end pipeline (train -> eval -> save) for selected model.

    Args:
        config: Configuration dictionary.
        logger: Logger instance.
        model_name: Model name selected from CLI.

    Returns:
        Dictionary with execution metadata.
    """
    selected_model = model_name.upper()
    logger.info(f"Pipeline task selected model: {selected_model}")

    if selected_model not in {"LSTM", "TRANSFORMER", "RNN"}:
        raise ValueError(f"Unsupported model: {selected_model}")

    data_config = config["data"]
    preprocess = data_config["preprocess"]
    paths_config = config["paths"]
    pipelines_config = config.get("pipelines", {})
    selected_pipeline_config = pipelines_config.get(selected_model, {})

    model_config = selected_pipeline_config.get("model", {})
    training_config = selected_pipeline_config.get("training", {})
    loss_config = selected_pipeline_config.get("loss", {})
    early_stopping_config = selected_pipeline_config.get("early_stopping", {})
    output_config = selected_pipeline_config.get("output", {})

    ds_dict = load_splits(
        splits=data_config["splits"],
        data_dir=paths_config["data_dir"],
        file_name_patterns=data_config["file_patterns"],
        exclude_sentiments=preprocess.get("exclude_sentiments"),
        exclude_topics=preprocess.get("exclude_topics"),
        sentiment_mapping=preprocess.get("sentiment_mapping"),
        sentiment_names=data_config["labels"]["sentiment"]["names"],
        topic_names=data_config["labels"]["topic"]["names"],
        encoding=preprocess.get("encoding", "utf-8"),
    )

    if selected_model == "LSTM":
        logger.info("Starting LSTM train -> eval -> save")
        lstm_pipline(
            ds_dict=ds_dict,
            n_epochs=training_config.get("epochs", 3),
            batch_size=training_config.get("batch_size", 16),
            min_freq=model_config.get("min_freq", 2),
            sent_alpha=loss_config.get("sentiment_label_smoothing", 0.0),
            topic_alpha=loss_config.get("topic_label_smoothing", 0.0),
            embedding_dim=model_config.get("embedding_dim", 300),
            hidden_dim=model_config.get("hidden_dim", 256),
            n_layers=model_config.get("n_layers", 2),
            dropout=model_config.get("dropout", 0.5),
            bidirectional=model_config.get("bidirectional", True),
            lr=training_config.get("learning_rate", 1e-3),
            weight_decay=training_config.get("weight_decay", 0.0),
            early_stopping_patience=early_stopping_config.get("patience", 10),
            early_stopping_verbose=early_stopping_config.get("verbose", True),
            early_stopping_delta=early_stopping_config.get("delta", 0.0),
            checkpoint_path=early_stopping_config.get(
                "checkpoint_path",
                "./outputs/lstm_multitask_best_model.pt",
            ),
            sentiment_names=data_config["labels"]["sentiment"]["names"],
            topic_names=data_config["labels"]["topic"]["names"],
            sentiment_report_path=output_config.get(
                "sentiment_report_path",
                "./logs/lstm_sentiment_classification_report.txt",
            ),
            topic_report_path=output_config.get(
                "topic_report_path",
                "./logs/lstm_topic_classification_report.txt",
            ),
            confusion_matrix_dir=output_config.get(
                "confusion_matrix_dir",
                "./images/lstm",
            ),
        )
        logger.info("LSTM pipeline completed")

    elif selected_model == "RNN":
        logger.info("Starting RNN train -> eval -> save")
        rnn_pipline(
            ds_dict=ds_dict,
            n_epochs=training_config.get("epochs", 30),
            batch_size=training_config.get("batch_size", 32),
            min_freq=model_config.get("min_freq", 2),
            embedding_dim=model_config.get("embedding_dim", 128),
            hidden_dim=model_config.get("hidden_dim", 96),
            n_layers=model_config.get("n_layers", 2),
            dropout=model_config.get("dropout", 0.35),
            bidirectional=model_config.get("bidirectional", True),
            rnn_type=model_config.get("rnn_type", "gru"),
            lr=training_config.get("learning_rate", 5e-4),
            weight_decay=training_config.get("weight_decay", 1e-4),
            sent_loss_weight=loss_config.get("sentiment_loss_weight", 1.0),
            topic_loss_weight=loss_config.get("topic_loss_weight", 1.5),
            grad_clip_norm=training_config.get("grad_clip_norm", 1.0),
            scheduler_factor=training_config.get("scheduler_factor", 0.5),
            scheduler_patience=training_config.get("scheduler_patience", 2),
            scheduler_min_lr=training_config.get("scheduler_min_lr", 1e-5),
            early_stopping_patience=early_stopping_config.get("patience", 5),
            early_stopping_verbose=early_stopping_config.get("verbose", True),
            early_stopping_delta=early_stopping_config.get("delta", 0.0),
            checkpoint_path=early_stopping_config.get(
                "checkpoint_path",
                "./outputs/rnn_multitask_best_model.pt",
            ),
            sentiment_names=data_config["labels"]["sentiment"]["names"],
            topic_names=data_config["labels"]["topic"]["names"],
            sentiment_report_path=output_config.get(
                "sentiment_report_path",
                "./logs/rnn_sentiment_classification_report.txt",
            ),
            topic_report_path=output_config.get(
                "topic_report_path",
                "./logs/rnn_topic_classification_report.txt",
            ),
            confusion_matrix_dir=output_config.get(
                "confusion_matrix_dir",
                "./images/rnn",
            ),
        )
        logger.info("RNN pipeline completed")

    else:
        raise NotImplementedError(
            f"Model '{selected_model}' is not implemented yet."
        )

    return {"model": selected_model}


def task_data_vis(config: dict, logger: logging.Logger) -> dict:
    """
    Load dataset splits and save distribution plots to image files.

    Args:
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Mapping from plot key to saved image path.
    """
    data_config = config["data"]
    data_dir = config["paths"]["data_dir"]
    file_name_patterns = data_config["file_patterns"]
    preprocess = data_config["preprocess"]

    images_dir = config["paths"].get(
        "images_dir",
        os.path.join(config["paths"].get("outputs_dir", "."), "images"),
    )

    ds_dict = load_splits(
        splits=data_config["splits"],
        data_dir=data_dir,
        file_name_patterns=file_name_patterns,
        exclude_sentiments=preprocess.get("exclude_sentiments"),
        exclude_topics=preprocess.get("exclude_topics"),
        sentiment_mapping=preprocess.get("sentiment_mapping"),
        sentiment_names=data_config["labels"]["sentiment"]["names"],
        topic_names=data_config["labels"]["topic"]["names"],
        encoding=preprocess.get("encoding", "utf-8"),
    )

    saved_paths = visualize_data(ds_dict=ds_dict, images_dir=images_dir)
    logger.info(f"Saved charts to: {images_dir}")
    return saved_paths


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Student Feedback Pipeline - Execute selected tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
      python run.py --task download                 # Download raw data only
            python run.py --task data-vis                 # Save data distribution charts
            python run.py --task pipeline --model LSTM    # Run full LSTM pipeline
        """
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["download", "data-vis", "pipeline"],
        help="Specify exactly one task to execute."
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["LSTM", "Transformer", "RNN"],
        help="Model name for --task pipeline.",
    )

    parser.add_argument(
        "--overwrite-data",
        action="store_true",
        help="Use with 'download' task to re-download files that already exist."
    )

    return parser.parse_args()


def get_tasks_to_run(args) -> list:
    """
    Determine which tasks to run based on arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        List of tasks to execute in order
    """
    all_tasks = ["download", "data-vis", "pipeline"]

    if args.task:
        if args.task not in all_tasks:
            raise ValueError(f"Invalid task: {args.task}")
        return [args.task]

    # Default: no task selected
    return []


def main():
    """
    Main execution function.

    Parses arguments and executes selected tasks in sequence.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Get configuration path
    config_path = os.path.join(
        os.path.dirname(__file__),
        "config.yml"
    )

    # Load configuration
    config = load_config(config_path)

    # Setup logging
    logger = setup_logging(
        log_dir=config["paths"]["logs_dir"],
        log_level=config["logging"].get("level", "INFO")
    )

    # Create necessary directories
    create_directories(config["paths"])
    logger.info("Pipeline started")

    # Get tasks to run
    try:
        tasks_to_run = get_tasks_to_run(args)
    except ValueError as e:
        logger.error(f"Invalid task specification: {e}")
        sys.exit(1)

    if args.task == "pipeline" and not args.model:
        logger.error("--model is required when --task pipeline")
        sys.exit(1)

    if args.task != "pipeline" and args.model:
        logger.error("--model is only allowed when --task pipeline")
        sys.exit(1)

    if not tasks_to_run:
        logger.info("No task selected. Use --task to run one pipeline task.")
        logger.info("Execution finished.")
        return {}

    logger.info(f"Tasks: {', '.join(tasks_to_run)}")

    selected_model = args.model
    if selected_model:
        logger.info(f"Selected model: {selected_model}")

    # Execute tasks in sequence
    task_results = {}
    task_functions = {
        "download": run_download,
        "data-vis": task_data_vis,
        "pipeline": task_run_pipeline,
    }

    try:
        for task_name in tasks_to_run:
            logger.info(f"Run task: {task_name}")
            task_func = task_functions[task_name]
            if task_name == "download":
                result = task_func(config, logger, overwrite=args.overwrite_data)
            elif task_name == "pipeline":
                result = task_func(config, logger, model_name=selected_model)
            else:
                result = task_func(config, logger)
            task_results[task_name] = result
            logger.info(f"Done: {task_name}")

        logger.info(f"Completed tasks: {', '.join(tasks_to_run)}")

        return task_results

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
