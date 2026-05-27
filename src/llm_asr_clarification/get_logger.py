import logging
from datetime import datetime

def get_logger(exp_name: str):
    logger = logging.getLogger(exp_name)

    now = datetime.now()
    # Format: Year-Month-Day Hour:Minute:Second
    formatted_string = now.strftime("%Y-%m-%d_%H:%M:%S")

    # IMPORTANT: set base threshold
    logger.setLevel(logging.DEBUG)

    # avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(f"./logs/{exp_name[:-3]}_{formatted_string}.log", mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s | %(name)s:\n%(message)s'))
    logger.addHandler(file_handler) 

    # logger.propagate = False #stops torch and huggingface and shit from logging

    return logger

