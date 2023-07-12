import argparse
import platform
from utils.common import show_lg
import datetime as dt

def parse_opt(known=False):
    parser = argparse.ArgumentParser(prog='PROG')
    subparsers = parser.add_subparsers(help='mulit-heads cls-command help')
    #! 多头分类模型，多数据源 - 异步训练
    parser_amm = subparsers.add_parser('amm', help='Asynchronous training and Mulit datalines and Mulit-heads help')
    parser_amm.add_argument('--net_name', type=str, default='', help='保存训练日志文件夹')
    parser_amm.add_argument('--data_path_1', type=str, default='pth_1', required=True, help='训练1-head 数据路径')
    parser_amm.add_argument('--data_path_2', type=str, default='pth_2', required=True, help='训练2-head 数据路径')
    parser_amm.add_argument('--data_path_3', type=str, default='pth_3', required=True, help='训练3-head 数据路径')
    parser_amm.add_argument('--head_indx', type=int, default=0, required=True, choices=[1,2,3],help='训练哪个头')
    parser_amm.add_argument('--resume_weights_1', type=str, default='', help='1-head 预训练完的模型')
    parser_amm.add_argument('--resume_weights_2', type=str, default='', help='2-head 预训练完的模型')
    parser_amm.add_argument('--save_weights_3', type=str, default='', help='3-head 保存的模型')
    parser_amm.add_argument('--log_name', type=str, default='train/amm', help='保存训练日志文件夹')
    #! 多头分类模型，单数据源 - 异步训练
    parser_asm = subparsers.add_parser('asm', help='Asynchronous training and Single dataline and Mulit-heads help')
    parser_asm.add_argument('--data_path_1', type=str, default='pth_1', required=True, help='1-head 数据路径')
    parser_asm.add_argument('--data_path_2', type=str, default='pth_2', required=True, help='2-head 数据路径')
    parser_asm.add_argument('--data_path_3', type=str, default='pth_3', required=True, help='3-head 数据路径')
    parser_asm.add_argument('--head_indx', type=int, default=0, required=True, help='训练哪个头')
    parser_asm.add_argument('--resume_weights_1', type=str, default='', help='1-head 预训练完的模型')
    parser_asm.add_argument('--resume_weights_2', type=str, default='', help='2-head 预训练完的模型')
    parser_asm.add_argument('--save_weights_3', type=str, default='', help='3-head 保存的模型')
    parser_asm.add_argument('--log_name', type=str, default='train/mha', help='保存训练日志文件夹')
    #! 多头分类模型，多数据源 - 同步训练 imgs VS labels
    parser_smm = subparsers.add_parser('smm', help='Synchronous training and Mulit datalines and Mulit-heads help')
    parser_smm.add_argument('--data_path_1', type=str, default='pth_1', required=True, help='1-head 数据路径')
    parser_smm.add_argument('--data_path_2', type=str, default='pth_2', required=True, help='2-head 数据路径')
    parser_smm.add_argument('--data_path_3', type=str, default='pth_3', required=True, help='3-head 数据路径')
    parser_smm.add_argument('--head_indx', type=int, default=0, required=True, help='训练哪个头')
    parser_smm.add_argument('--resume_weights_1', type=str, default='', help='1-head 保存模型和训练完的模型')
    parser_smm.add_argument('--resume_weights_2', type=str, default='', help='2-head 保存模型和训练完的模型')
    parser_smm.add_argument('--save_weights_3', type=str, default='', help='3-head 保存的模型')
    parser_smm.add_argument('--log_name', type=str, default='train/mhs', help='保存训练日志文件夹')
    #! 多头分类模型，单数据源 - 同步训练 img VS labels
    parser_ssm = subparsers.add_parser('ssm', help='Synchronous training and Single dataline and Mulit-heads help')
    parser_ssm.add_argument('--data_path_1', type=str, default='pth_1', required=True, help='1-head 数据路径')
    parser_ssm.add_argument('--data_path_2', type=str, default='pth_2', required=True, help='2-head 数据路径')
    parser_ssm.add_argument('--data_path_3', type=str, default='pth_3', required=True, help='3-head 数据路径')
    parser_ssm.add_argument('--head_indx', type=int, default=0, required=True, help='训练哪个头')
    parser_ssm.add_argument('--resume_weights_1', type=str, default='', help='1-head 保存模型和训练完的模型')
    parser_ssm.add_argument('--resume_weights_2', type=str, default='', help='2-head 保存模型和训练完的模型')
    parser_ssm.add_argument('--save_weights_3', type=str, default='', help='3-head 保存的模型')
    parser_ssm.add_argument('--log_name', type=str, default='train/mhs', help='保存训练日志文件夹')

    #! single-cls-head
    parser_sh = subparsers.add_parser('sh', help='single-head training help')
    parser_sh.add_argument('--data_path', type=str, default='pth_1', required=True, help='数据路径')
    parser_sh.add_argument("--net",type=str,default='yolov5_62cls_st',help='选择训练的网络')
    parser_sh.add_argument('--resume_weights', type=str, default='', help='1-head 保存的模型')
    parser_sh.add_argument('--n_cls', type=int, nargs='+',default=2, help='mulit-heads: num_classes')
    parser_sh.add_argument('--log_name', type=str, default='sh', help='保存训练日志文件夹')

    #! kd-cls
    parser_kd = subparsers.add_parser('kd', help='kd single-head model')
    parser_kd.add_argument('--data_path', type=str, default='pth_1', required=True, help='数据路径')
    parser_kd.add_argument("--teacher_model",type=str,default='resnet18',help='teacher网络名字')
    parser_kd.add_argument('--pre_weights', type=str, default='', help='teacher训练模型')
    parser_kd.add_argument("--net",type=str,default='yolov5_62cls_st',help='选择训练的网络')
    parser_kd.add_argument('--n_cls', type=int, nargs='+',default=2, help='mulit-heads: num_classes')
    parser_kd.add_argument('--log_name', type=str, default='sh', help='保存训练日志文件夹')
    #! meta learning cls
    parser_mtcls = subparsers.add_parser('mtcls', help='kd single-head model')
    parser_mtcls.add_argument('--data_path', type=str, default='pth_1', required=True, help='数据路径')
    parser_mtcls.add_argument("--teacher_model",type=str,default='resnet18',help='teacher网络名字')
    parser_mtcls.add_argument('--pre_weights', type=str, default='', help='teacher训练模型')
    parser_mtcls.add_argument("--net",type=str,default='yolov5_62cls_st',help='选择训练的网络')
    parser_mtcls.add_argument('--n_cls', type=int, nargs='+',default=2, help='mulit-heads: num_classes')
    parser_mtcls.add_argument('--log_name', type=str, default='sh', help='保存训练日志文件夹')
    # train-args and Shared parameters
    parser.add_argument('--resume', type=str, default='', help="预训练模型")
    parser.add_argument('--input_size', type=int, nargs='+', default=[224,224], help="输入图像大小")
    parser.add_argument('--epochs', type=int, default=3, help="训练epochs")
    parser.add_argument('--batch_size', type=int, default=32, help='batch-size大小')
    parser.add_argument('--n_cuda', type=str, nargs='+', default=['0'], help='gpu id')
    parser.add_argument('--data2index', type=int, nargs='+', default=-2, help='数据源对应的label')
    
    parser.add_argument('--workers', type=int, default=2, help='训练模型', choices=[0,1,2,3])

    parser.add_argument('--init_method', type=int,default=0, help='learning ratio')
    parser.add_argument('--lr', type=float,default=1e-5, help='learning ratio')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    # default
    # parser.add_argument('--flag', type=int, default=0, help='训练模型', choices=[0, 1, 2, 3])
    opts = parser.parse_known_args()[0] if known else parser.parse_args()
    opts.mode_Lst = ['train', 'val','test']
    
    opts.early_lst = {"acc":0.99,'loss':0.002} # '早停阈值'

    if 'win' in platform.platform().lower():
        show_lg('using window and worker be 0')
        opts.workers=0
    return opts

if __name__ == '__main__':
    print(parse_opt())
