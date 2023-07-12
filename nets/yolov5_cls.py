import torch.nn as nn

from nets.utils.layer_tools import C3,Conv,Classify 
from nets.utils.struct_cfg import yolov5_cfg

class yolov5_cls_st(nn.Module):
    def __init__(self,cfg:dict):
        super(yolov5_cls_st, self).__init__()

        feature_map = nn.ModuleList()
        feature_map_cfg = cfg['feature_map']
        feature_map.append(Conv(*feature_map_cfg[0][1]))    
        feature_map.append(Conv(*feature_map_cfg[1][1])) 
        feature_map.extend([C3(*feature_map_cfg[2][1])]*feature_map_cfg[2][0][0]) 
        feature_map.append(Conv(*feature_map_cfg[3][1])) 
        feature_map.extend([C3(*feature_map_cfg[4][1])]*feature_map_cfg[4][0][0]) 
        feature_map.append(Conv(*feature_map_cfg[5][1])) 
        feature_map.extend([C3(*feature_map_cfg[6][1])]*feature_map_cfg[6][0][0]) 
        feature_map.append(Conv(*feature_map_cfg[7][1])) 
        feature_map.extend([C3(*feature_map_cfg[8][1])]*feature_map_cfg[8][0][0]) 
        self.feature_ =  nn.Sequential(*feature_map)

        self.cls = nn.Sequential()
        self.classify_cfg  = classify_cfg = cfg['classify']
        self.cls.add_module(f'cls_1',Classify(*classify_cfg[1]))
    
    def forward(self,x):
        x = self.feature_(x)
        out1 = self.cls(x)
        out1 = out1.view([-1,8,int(self.classify_cfg/7)])
        return out1




#~ s meaning st-struct
def yolov5n_cls(num_cls1):
    cfg = yolov5_cfg(num_classes1=num_cls1,num_classes2=0,num_classes3=0).yolov5_n()
    return yolov5_cls_st(cfg)

def yolov5s_cls(num_cls1):
    cfg = yolov5_cfg(num_classes1=num_cls1,num_classes2=0,num_classes3=0).yolov5_s()
    return yolov5_cls_st(cfg)

def yolov5l_cls(num_cls1):
    cfg = yolov5_cfg(num_classes1=num_cls1,num_classes2=0,num_classes3=0).yolov5_l()
    return yolov5_cls_st(cfg)
     
def yolov5m_cls(num_cls1):
    cfg = yolov5_cfg(num_classes1=num_cls1,num_classes2=0,num_classes3=0).yolov5_m()
    return yolov5_cls_st(cfg)




