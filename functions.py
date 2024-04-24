import os
import re


def get_highest_epoch():
    L_EPOCHS = []
    REGEX = r"model_epoch_(\d+).pth"

    files = os.listdir("model")
    for file in files:
        match = re.fullmatch(REGEX, file)
        if match:
            L_EPOCHS.append(match.group(1))

    return max(L_EPOCHS)
