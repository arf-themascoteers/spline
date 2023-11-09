import shutil
import os


def clear_all():
    shutil.rmtree("checkpoints")
    shutil.rmtree("lightning_logs")
    shutil.rmtree("results")

    os.mkdir("checkpoints")
    os.mkdir("lightning_logs")
    os.mkdir("results")