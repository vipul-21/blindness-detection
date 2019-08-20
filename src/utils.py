from datetime import datetime
import torch

def getDevice():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model_stamp():
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')