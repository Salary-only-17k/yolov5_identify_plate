# from lib.nets.resnet_cn import resnet50_3head_cnn,resnet50_3head_cnl,resnet50_2head_cnn,resnet50_2head_cnl
# from lib.nets.resnet_dw import resnet50_3head_dwn,resnet50_3head_dwl,resnet50_2head_dwn,resnet50_2head_dwl
# from lib.nets.resnet_dp import resnet50_3head_dpn,resnet50_3head_dpl,resnet50_2head_dpn,resnet50_2head_dpl
# from lib.nets.experiment.densenet_mulitheread import DenseNet_3head_l,DenseNet_2head_l
# from lib.nets.yolov5_62cls import yolov5n_cls_s,yolov5s_cls_s,yolov5l_cls_s,yolov5m_cls_s, \
#                                   yolov5n_cls_m,yolov5s_cls_m,yolov5l_cls_m,yolov5m_cls_m, \
#                                   yolov5n_cls_l,yolov5s_cls_l,yolov5l_cls_l,yolov5m_cls_l
# from lib.nets.yolov5_50cls import yolov5s_50cls,yolov5n_50cls
# from lib.nets.efficientnet_gai import efficientnet_v2_s3h,efficientnet_b0_3h,efficientnet_b1_3h,efficientnet_v2_l3h


# all_net = {"res50_3h_n":[resnet50_3head_cnn,resnet50_3head_dpn,resnet50_3head_dwn],
#             "res50_3h_l": [resnet50_3head_cnl,resnet50_3head_dpl,resnet50_3head_dwl],
#             "res50_2h_n":[resnet50_2head_cnn,resnet50_2head_dpn,resnet50_2head_dwn],
#             "res50_2h_l":[resnet50_2head_cnl,resnet50_2head_dpl,resnet50_2head_dwl],
#             "des161_3h_l":DenseNet_3head_l,
#             "des50_2h_l":DenseNet_2head_l,
#             "yolov5_62cls_s":[yolov5n_cls_s,yolov5s_cls_s,yolov5l_cls_s,yolov5m_cls_s,],
#             "yolov5_62cls_m":[yolov5n_cls_m,yolov5s_cls_m,yolov5l_cls_m,yolov5m_cls_m,],
#             "yolov5_62cls_l":[yolov5n_cls_l,yolov5s_cls_l,yolov5l_cls_l,yolov5m_cls_l],
#             "yolov5_50cls":[yolov5s_50cls,yolov5n_50cls],
#             "efficientnet":[efficientnet_v2_s3h,efficientnet_b0_3h,efficientnet_b1_3h,efficientnet_v2_l3h]}

from nets.yolov5_cls import yolov5n_cls, yolov5s_cls, yolov5l_cls,yolov5m_cls
from nets.resnet_cls import resnet18,resnet34

all_net = {"yolovv5_cls":[yolov5n_cls, yolov5s_cls, yolov5l_cls,yolov5m_cls,],
            "resnet":[resnet18,resnet34,],
            
          }
