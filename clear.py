import shutil
import os


def clear_all():
    for folder in ["checkpoints", "lightning_logs", "results"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            os.mkdir(folder)
