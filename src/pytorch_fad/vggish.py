import torch
import torch.nn as nn
import torch.nn.functional as F


def get_vggish_model(device=None):
    model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    if device is not None:
        device = torch.device(device) #dirty hack since the model moves itself to gpu if it exists anyways, oh well
        model.to(device)
        model.device = device
        torch.cuda.empty_cache()
    model.eval()
    model.postprocess = False
    model.preprocess = False
    model.embeddings[5] = nn.Identity()
    return model
