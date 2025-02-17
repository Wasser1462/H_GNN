# Author: zyw
# Date: 2024-11-01
# Description: 
import torch

def print_model_parameters(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    
    if isinstance(model, dict) and 'state_dict' in model:
        model = model['state_dict']
    elif hasattr(model, 'state_dict'):
        model = model.state_dict()
    print("Model Parameters:\n")
    for name, param in model.items():
        print(f"Layer: {name} | Size: {param.size()} | Requires Grad: {param.requires_grad}")
        #print(f"Parameters:\n{param}\n")
    
model_path ='/data1/zengyongwang/TeleSpeech-ASR/wenet_representation/exp/d2v2_ark_conformer_cantonese/final.pt'
print_model_parameters(model_path)