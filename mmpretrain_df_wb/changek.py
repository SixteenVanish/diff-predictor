import torch
import collections

def change_key_names(checkpoint_path, new_checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    new_checkpoint = {}
    for key, value in checkpoint.items():
        new_key = key + '.'
        new_checkpoint[new_key] = value
        if isinstance(value, collections.OrderedDict):
            for key0, value0 in value.items():
                if key0[:7] == 'module.':
                    key0 = key0[7:]
                    key0 = new_key + key0
                # print(key0)
                new_checkpoint[key0] = value0
    torch.save(new_checkpoint, new_checkpoint_path)

if __name__ == '__main__':
    change_key_names(checkpoint_path='logs/vitb16.224-8xb256-100e_in1k/jepa-ep100.pth.tar', 
                     new_checkpoint_path='logs/vitb16.224-8xb256-100e_in1k/jepa-ep100_.pth.tar')