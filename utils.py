import logging

def setup_logging(log_path='results/logs/train.log'):
    logging.basicConfig(filename=log_path, level=logging.INFO)
    return logging.getLogger()
