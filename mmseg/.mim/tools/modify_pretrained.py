from collections import OrderedDict

import torch

state_dict = torch.load('/home/gis/wzq/code/0-experiment-platform/pretrained/tw_decoder_200e.pth',
                            map_location='cuda:0')
encoder_dict = OrderedDict()
decoder_dict = OrderedDict()
for key in state_dict.keys(): # 'module.'
    if 'Tdec' not in key and 'convtail' not in key and 'clean' not in key and 'active' not in key:
        encoder_dict[key[7:]] = state_dict[key]
    else:
        decoder_dict[key[7:]] = state_dict[key]
torch.save(encoder_dict,
           '/pretrained/tw_backbone.pth')
torch.save(decoder_dict,
           '/pretrained/tw_decoder_200e.pth')
