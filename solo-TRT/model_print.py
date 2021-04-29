from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import struct
import torch


torch.set_printoptions(profile="full")
def print_mask_feat_head():
    checkpoint_file = '/home/sqc/learn/code/segmentation/SOLO/checkpoints/SOLOv2_R101_3x.pth'
    model = torch.load(checkpoint_file,map_location=torch.device('cuda:0'))
    weights = model['state_dict']
    f=open('/home/sqc/learn/code/segmentation/SOLO/SQC/logs/mask_feat_head_gn.txt','w' )
    f.write('backbone.layer1.0.bn2.weight:{} \nbias:{} \nrunning_mean:{} \nrunning_var:{}'.format(
        weights['backbone.layer1.0.bn2.weight'],
        weights['backbone.layer1.0.bn2.bias'],
        weights['backbone.layer1.0.bn2.running_mean'],
        weights['backbone.layer1.0.bn2.running_var']
    ) )

    cate_lname = 'bbox_head.cate_convs.'
    kernel_lname = 'bbox_head.kernel_convs.'
    for i in range(4):
        cate_conv=cate_lname+str(i)+'.conv.weight'
        cate_gnw=cate_lname+str(i)+'.gn.weight'
        cate_gnb=cate_lname+str(i)+'.gn.bias'
        f.write('\n====================================convs_all_levels:{}====================================='.format(i))
        f.write('\n {}:{}\n {}:{}\n {}:{}\n '.format(cate_conv,weights[cate_conv].shape, cate_gnw,weights[cate_gnw].shape, cate_gnb,weights[cate_gnb].shape))
        f.write('\n {}:{}\n {}:{}\n '.format(cate_gnw,weights[cate_gnw], cate_gnb,weights[cate_gnb]))

        kernel_conv=kernel_lname+str(i)+'.conv.weight'
        kernel_gnw=kernel_lname+str(i)+'.gn.weight'
        kernel_gnb=kernel_lname+str(i)+'.gn.bias'        
        f.write('\n {}:{}\n {}:{}\n {}:{}\n '.format(kernel_conv,weights[kernel_conv].shape, kernel_gnw,weights[kernel_gnw].shape, kernel_gnb,weights[kernel_gnb].shape))
        f.write('\n {}:{}\n {}:{}\n '.format(kernel_gnw,weights[kernel_gnw], kernel_gnb,weights[kernel_gnb]))


def gen_weights():

    checkpoint_file = '/home/sqc/learn/code/segmentation/SOLO/checkpoints/SOLOv2_R101_3x.pth'
    # build the model from a config file and a checkpoint file
    model = torch.load(checkpoint_file,map_location=torch.device('cuda:0'))
    for item in model.items():
        print('{} {}'.format(item[0],type(item[1]) ) )

    weights = model['state_dict']
    print('weights:{} type:{}'.format(len(weights), type(weights)))
    for k,v in weights.items():
        print(k)
    meta = model['meta']
    for k,v in meta.items():
        print('k:{} v:{}'.format) 



def print_model():
    config_file = '/home/sqc/learn/code/segmentation/SOLO/configs/solov2/solov2_r101_fpn_8gpu_3x.py'
    # download the checkpoint from model zoo and put it in `checkpoints/`
    checkpoint_file = '/home/sqc/learn/code/segmentation/SOLO/checkpoints/SOLOv2_R101_3x.pth'
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    print(type( model) )

    # f1=open('/home/sqc/learn/code/segmentation/SOLO/checkpoints/SOLOv2_R101_3x.wts','w')
    # f1.write('{}\n'.format(len(model.state_dict().keys() )))
    f2=open('/home/sqc/learn/code/segmentation/SOLO/SQC/logs/SOLOv2_R101_3x.log','w')


    for k,v in model.state_dict().items():
        f2.write('k:{} shape:{}\n'.format(k,v.shape) )
        # vr = v.reshape(-1).cpu().numpy()        
        # f1.write('{} {}'.format(k, len(vr)))
        # for vv in vr:
        #     f1.write(' ')
        #     f1.write(struct.pack('>f', float(vv)).hex())
        # f1.write('\n')

    # for i,(k,v) in enumerate(model.state_dict().items()):
    #     if i%100 ==0:
    #         f2=open('/home/sqc/learn/code/segmentation/SOLO/checkpoints/SOLOv2_R101_3x.log_'+str(i),'w')
    #         f2.write('{}\n'.format(len(model.state_dict().keys() )))
    #     vr = v.reshape(-1).cpu().numpy()        
    #     f2.write('k:{} len:{} shape:{}'.format(k, len(vr), v.shape ))
    #     for vv in vr:
    #         f2.write(' ')
    #         # f2.write(struct.pack('>f', float(vv)).hex())  #101
    #         f2.write( str(float(vv)))   #100
    #     f2.write('\n')
    # f2.write('\n')
    f2.close()

if __name__ =='__main__':
    # print_model()
    # gen_weights()
    print_mask_feat_head()
