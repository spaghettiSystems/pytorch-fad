import torch
import torch.nn as nn
import torch.nn.functional as F

def get_vggish_model():
    model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    model.eval()
    model.postprocess = False
    model.preprocess = False
    model.embeddings[5] = nn.Identity()
    return model