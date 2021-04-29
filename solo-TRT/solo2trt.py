import tensorrt as trt
import numpy as np
import torch
import pycuda.autoinit
import pycuda.driver as cuda
import cv2


logger = trt.Logger(trt.Logger.WARNING)
class ModelData(object):
    INPUT_NAME = "Date"
    INPUT_SHAPE = (1, 3, )
    OUTPUT_NAME = "Prob"
    DTYPE = trt.float32


def addBatchNorm2d(network, weights, input_tensor, lname, eps):
    m0 = weights[lname+'.running_mean'].numpy()
    v0 = weights[lname+'.running_var'].numpy()
    g0 = weights[lname+'.weight'].numpy()
    b0 = weights[lname+'.bias'].numpy()

    scale = g0 / np.sqrt(v0 + eps)
    shift = b0 - scale * m0
    power = np.ones(len(g0), dtype = np.float32)
    bnLayer = network.add_scale(input_tensor, trt.ScaleMode.CHANNEL, shift, scale, power)
    return bnLayer

def bottleneck(network, weights, input_tensor, inch, outch, stride, lname):
    conv1_w = weights[lname+'conv1.weight'].numpy()
    # conv1_b = weights[lname+'conv1.bias'].numpy()
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=outch, kernel_shape=(1,1), kernel=conv1_w)

    bn1 = addBatchNorm2d(network, weights, conv1.get_output(0), lname+'bn1', 1e-5)
    relu1 = network.add_activation(input=bn1.get_output(0), type=trt.ActivationType.RELU)

    conv2_w = weights[lname+'conv2.weight'].numpy()
    # conv2_b = weights[lname+'conv2.bias'].numpy()
    conv2 = network.add_convolution(input=relu1.get_output(0), num_output_maps=outch, kernel_shape=(3,3), kernel=conv2_w)
    conv2.stride = (stride,stride)
    conv2.padding = (1,1)
    bn2 = addBatchNorm2d(network, weights, conv2.get_output(0), lname+'bn2', 1e-5)
    relu2 = network.add_activation(input=bn2.get_output(0), type=trt.ActivationType.RELU)

    conv3_w = weights[lname+'conv3.weight'].numpy()
    # conv3_b = weights[lname+'conv3.bias'].numpy()
    conv3 = network.add_convolution(input=relu2.get_output(0), num_output_maps=outch*4, kernel_shape=(1,1), kernel=conv3_w)
    bn3 = addBatchNorm2d(network, weights, conv3.get_output(0), lname+'bn3', 1e-5)

    if (stride != 1 or inch != outch*4):
        conv4_w = weights[lname+'downsample.0.weight'].numpy()
        # conv4_b = weights[lname+'downsample.0.bias'].numpy()
        conv4 = network.add_convolution(input=input_tensor, num_output_maps=outch*4, kernel_shape=(2, 2), kernel=conv4_w)
        conv4.stride = (stride, stride)
        bn4 = addBatchNorm2d(network, weights, conv4.get_output(0), lname+'downsample.1', 1e-5)
        ew1 = network.add_elementwise(bn4.get_output(0), bn3.get_output(0),trt.ElementWiseOperation.SUM)
    else:
        ew1 = network.add_elementwise(input_tensor, bn3.get_output(0), trt.ElementWiseOperation.SUM)

    relu3 = network.add_activation(ew1.get_output(0), trt.ActivationType.RELU)
    return relu3

def make_res_layer(network, weights, last_layer, block, in_channels, out_channels, stride, num_blocks, expansions, lname):
    # lname 'backbone.layer1.'
    layer=last_layer
    for i in range(num_blocks):
        layer = block(network, weights, layer.get_output(0), in_channels, out_channels, stride, lname+str(i)+'.')
        in_channels = out_channels*expansions
        stride=1
    return layer

def build_resnet(network, weights, last_layer, res_num):
    arch_settings = {
        50: (bottleneck, (3, 4, 6, 3)),
        101: (bottleneck, (3, 4, 23, 3)),
        152: (bottleneck, (3, 8, 36, 3))
    }
    in_channels=64
    expansions=4
    block, block_stages= arch_settings[res_num]
    outs=[]
    layer=last_layer
    stride=1
    for i, num_blocks in enumerate(block_stages):
        out_channels=64*2**i                                                                                    #'backbone.layer1.'
        layer=make_res_layer(network, weights, layer, block, in_channels, out_channels, stride, num_blocks, expansions, 'backbone.layer'+str(i+1)+'.')
        outs.append(layer)
        in_channels = out_channels*expansions
        stride=2
    return outs

def build_fpn_lateral(network, weights, input_tensors):
    outs=[]
    for i,tensor in enumerate(input_tensors):
        lname='neck.lateral_convs.'+str(i)  #neck.lateral_convs.0
        l_w = weights[lname+'.conv.weight'].numpy()
        l_b = weights[lname+'.conv.bias'].numpy()
        l = network.add_convolution(input=tensor.get_output(0), num_output_maps=256, kernel_shape=(1, 1), kernel=l_w, bias=l_b)
        l.stride = (1, 1)
        outs.append(l)
    for i in range( len(outs)-1,0,-1):
        resize_layer = network.add_resize(outs[i].getOutput(0))
        resize_layer.scales = [1,1,2,2]
        resize_layer.resize_mode =trt.ResizeMode.NEAREST
        resize_layer.align_corners = False
        add_layer = network.add_elementwise(outs[i-1].getOutput(0), resize_layer.getOutput(0), op=trt.ElementWiseOperation.SUM)
        outs[i-1] = add_layer
    return outs

def build_fpn_conv(network, weights, input_tensors):
    outs=[]
    for i,tensor in enumerate(input_tensors):
        lname='neck.fpn_convs.'+str(i)  #neck.lateral_convs.0
        l_w = weights[lname+'.conv.weight'].numpy()
        l_b = weights[lname+'.conv.bias'].numpy()
        fpn_layer = network.add_convolution(input=tensor.get_output(0),
            num_output_maps=256, kernel_shape=(3, 3), kernel=l_w, bias=l_b)
        outs.append(fpn_layer)
    # outs[-1]
    pool_layer=network.add_pooling(outs[len(outs)-1].getOutput(0), type=trt.PoolingType.MAX, window_size=(1,1))
    pool_layer.stride = (2,2)
    outs.add(pool_layer)
    return outs    

def resize_and_split(network, input_tensors):
    tensor0 = input_tensors[0]
    resize_layer0=network.add_resize(tensor0.get_output(0))
    resize_layer0.scales = [1,1,0.5,0.5]
    resize_layer0.resize_mode = trt.ResizeMode.LINEAR
    input_tensors[0] = resize_layer0

    tensor4 = input_tensors[4]
    resize_layer4=network.add_resize(tensor4.get_output(0))
    resize_layer4.scales = [1,1,2,2]
    resize_layer0.resize_mode = trt.ResizeMode.LINEAR
    input_tensors[4] = resize_layer4
    return input_tensors




def solov2_head(network, weights, input_tensors):
    tensor0 = input_tensors[0]
    resize_layer0=network.add_resize(tensor0.get_output(0))
    resize_layer0.scales = [1,1,0.5,0.5]
    resize_layer0.resize_mode = trt.ResizeMode.LINEAR

    


def solov2_network(network, weights):
    f=open('/home/sqc/learn/code/segmentation/SOLO/SQC/logs/solov2_network.txt','w')

    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)
    # blackbone: ResNet101
    conv1_w = weights['backbone.conv1.weight'].numpy()
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=64, kernel_shape=(7,7), kernel=conv1_w)
    conv1.stride = (2,2)
    conv1.padding = (3,3)
    bn1 = addBatchNorm2d(network, weights, conv1.get_output(0), 'backbone.bn1', 1e-5)
    relu1 = network.add_activation(input=bn1.get_output(0), type=trt.ActivationType.RELU)
    pool1 = network.add_pooling(input=relu1.get_output(0), type=trt.PoolingType.MAX, window_size=(3,3))
    pool1.stride = (2, 2)
    pool1.padding = (1, 1)
    outs = build_resnet(network, weights, pool1, 101)

    outs = build_fpn_lateral(network, weights, outs)
    outs = build_fpn_conv(network, weights, outs)

    
    for out in outs:
        f.write('out:{} {}\n'.format(out,out.shape))

    # NCK: FPN
    # lateral_convs
    fpn_conv0_w = weights['neck.lateral_convs.0.conv.weight'].numpy()
    fpn_conv0_b = weights['neck.lateral_convs.0.conv.bias'].numpy()
    fpn_conv0 = network.add_convolution(input=x.get_output(0), num_output_maps=256, kernel_shape=(1, 1), kernel=fpn_conv0_w, bias=fpn_conv0_b)
    fpn_conv0.stride = (1, 1)

    fpn_conv1_w = weights['neck.lateral_convs.1.conv.weight'].numpy()
    fpn_conv1_b = weights['neck.lateral_convs.1.conv.bias'].numpy()
    fpn_conv1 = network.add_convolution(input=fpn_conv0.get_output(0), num_output_maps=256, kernel_shape=(1, 1),
                                        kernel=fpn_conv1_w, bias=fpn_conv1_b)
    fpn_conv1.stride = (1, 1)

    fpn_conv2_w = weights['neck.lateral_convs.2.conv.weight'].numpy()
    fpn_conv2_b = weights['neck.lateral_convs.2.conv.bias'].numpy()
    fpn_conv2 = network.add_convolution(input=fpn_conv1.get_output(0), num_output_maps=256, kernel_shape=(1, 1),
                                        kernel=fpn_conv2_w, bias=fpn_conv2_b)
    fpn_conv2.stride = (1, 1)

    fpn_conv3_w = weights['neck.lateral_convs.3.conv.weight'].numpy()
    fpn_conv3_b = weights['neck.lateral_convs.3.conv.bias'].numpy()
    fpn_conv3 = network.add_convolution(input=fpn_conv2.get_output(0), num_output_maps=256, kernel_shape=(1, 1),
                                        kernel=fpn_conv3_w, bias=fpn_conv3_b)
    fpn_conv3.stride = (1, 1)

    # fpn_convs
    fpn_conv4_w = weights['neck.fpn_convs.0.conv.weight'].numpy()
    fpn_conv4_b = weights['neck.fpn_convs.0.conv.bias'].numpy()
    fpn_conv4 = network.add_convolution(input=fpn_conv3.get_output(0), num_output_maps=256, kernel_shape=(3, 3),
                                        kernel=fpn_conv4_w, bias=fpn_conv4_b)
    fpn_conv4.stride = (1, 1)
    fpn_conv4.padding = (1, 1)

    fpn_conv5_w = weights['neck.fpn_convs.1.conv.weight'].numpy()
    fpn_conv5_b = weights['neck.fpn_convs.1.conv.bias'].numpy()
    fpn_conv5 = network.add_convolution(input=fpn_conv4.get_output(0), num_output_maps=256, kernel_shape=(1, 1),
                                        kernel=fpn_conv5_w, bias=fpn_conv5_b)
    fpn_conv5.stride = (1, 1)
    fpn_conv5.padding = (1, 1)

    fpn_conv6_w = weights['neck.fpn_convs.2.conv.weight'].numpy()
    fpn_conv6_b = weights['neck.fpn_convs.2.conv.bias'].numpy()
    fpn_conv6 = network.add_convolution(input=fpn_conv5.get_output(0), num_output_maps=256, kernel_shape=(1, 1),
                                        kernel=fpn_conv6_w, bias=fpn_conv6_b)
    fpn_conv6.stride = (1, 1)
    fpn_conv6.padding = (1, 1)

    fpn_conv7_w = weights['neck.fpn_convs.3.conv.weight'].numpy()
    fpn_conv7_b = weights['neck.fpn_convs.3.conv.bias'].numpy()
    fpn_conv7 = network.add_convolution(input=fpn_conv6.get_output(0), num_output_maps=256, kernel_shape=(1, 1),
                                        kernel=fpn_conv7_w, bias=fpn_conv7_b)
    fpn_conv7.stride = (1, 1)
    fpn_conv7.padding = (1, 1)


    # mask_feat_head
    # mask_feat_head.convs_all_leaves
    # 0
    mask_conv0_w = weights['mask_feat_head.convs_all_levels.0.conv0.conv.weight'].numpy()
    mask_conv0 = network.add_convolution(input=fpn_conv7.get_output(0), num_output_maps=128, kernel_shape=(3, 3),
                                        kernel=mask_conv0_w)
    mask_conv0.stride = (1, 1)
    mask_conv0.padding = (1, 1)
    mask_gn0 = addBatchNorm2d(network, weights, mask_conv0.get_Output(0), 'mask_feat_head.convs_all_levels.0.conv0.gn', 1e-5)
    mask_relu0 = network.add_activation(input=mask_gn0.get_output(0), type=trt.ActivationType.RELU)

    # 1
    mask_conv1_w = weights['mask_feat_head.convs_all_levels.1.conv0.conv.weight'].numpy()
    mask_conv1 = network.add_convolution(input=mask_relu0.get_output(0), num_output_maps=128, kernel_shape=(3, 3),
                                         kernel=mask_conv1_w)
    mask_conv1.stride = (1, 1)
    mask_conv1.padding = (1, 1)
    mask_gn1 = addBatchNorm2d(network, weights, mask_conv1.get_Output(0), 'mask_feat_head.convs_all_levels.1.conv0.gn', 1e-5)
    mask_relu1 = network.add_activation(input=mask_gn1.get_output(0), type=trt.ActivationType.RELU)

    mask_deconv0_w = weights['mask_feat_head.convs_all_levels.1.upsample0.weight'].numpy()
    mask_deconv0 = network.add_deconvolution(input=mask_relu1.get_output(0), num_output_maps=128, kernel_shape=(2, 2),
                                         kernel=mask_deconv0_w)
    mask_deconv0.stride = (2, 2)
    mask_deconv0.groups = 129

    # 2
    mask_conv2_w = weights['mask_feat_head.convs_all_levels.2.conv0.conv.weight'].numpy()
    mask_conv2 = network.add_convolution(input=mask_deconv0.get_output(0), num_output_maps=128, kernel_shape=(3, 3),
                                         kernel=mask_conv2_w)
    mask_conv2.stride = (1, 1)
    mask_conv2.padding = (1, 1)
    mask_gn2 = addBatchNorm2d(network, weights, mask_conv2.get_Output(0), 'mask_feat_head.convs_all_levels.2.conv0.gn', 1e-5)
    mask_relu2 = network.add_activation(input=mask_gn2.get_output(0), type=trt.ActivationType.RELU)

    mask_deconv1_w = weights['mask_feat_head.convs_all_levels.2.upsample0.weight'].numpy()
    mask_deconv1 = network.add_deconvolution(input=mask_relu2.get_output(0), num_output_maps=128, kernel_shape=(2, 2),
                                             kernel=mask_deconv1_w)
    mask_deconv1.stride = (2, 2)
    mask_deconv1.groups = 128

    mask_conv3_w = weights['mask_feat_head.convs_all_levels.2.conv1.conv.weight'].numpy()
    mask_conv3 = network.add_convolution(input=mask_deconv1.get_output(0), num_output_maps=128, kernel_shape=(3, 3),
                                         kernel=mask_conv3_w)
    mask_conv3.stride = (1, 1)
    mask_conv3.padding = (1, 1)
    mask_gn3 = addBatchNorm2d(network, weights, mask_conv3.get_Output(0), 'mask_feat_head.convs_all_levels.2.conv1.gn', 1e-5)
    mask_relu3 = network.add_activation(input=mask_gn3.get_output(0), type=trt.ActivationType.RELU)

    mask_deconv2_w = weights['mask_feat_head.convs_all_levels.2.upsample1.weight'].numpy()
    mask_deconv2 = network.add_deconvolution(input=mask_relu3.get_output(0), num_output_maps=128, kernel_shape=(2, 2),
                                             kernel=mask_deconv2_w)
    mask_deconv2.stride = (2, 2)
    mask_deconv2.groups = 128

    # 3
    mask_conv4_w = weights['mask_feat_head.convs_all_levels.3.conv0.conv.weight'].numpy()
    mask_conv4 = network.add_convolution(input=mask_deconv2.get_output(0), num_output_maps=128, kernel_shape=(3, 3),
                                         kernel=mask_conv4_w)
    mask_conv4.stride = (1, 1)
    mask_conv4.padding = (1, 1)
    mask_gn4 = addBatchNorm2d(network, weights, mask_conv4.get_Output(0), 'mask_feat_head.convs_all_levels.3.conv0.gn', 1e-5)
    mask_relu4= network.add_activation(input=mask_gn4.get_output(0), type=trt.ActivationType.RELU)

    mask_deconv3_w = weights['mask_feat_head.convs_all_levels.3.upsample0.weight'].numpy()
    mask_deconv3 = network.add_deconvolution(input=mask_relu4.get_output(0), num_output_maps=128, kernel_shape=(2, 2),
                                             kernel=mask_deconv3_w)
    mask_deconv3.stride = (2, 2)
    mask_deconv3.groups = 128

    mask_conv5_w = weights['mask_feat_head.convs_all_levels.3.conv1.conv.weight'].numpy()
    mask_conv5 = network.add_convolution(input=mask_deconv3.get_output(0), num_output_maps=128, kernel_shape=(3, 3),
                                         kernel=mask_conv5_w)
    mask_conv5.stride = (1, 1)
    mask_conv5.padding = (1, 1)
    mask_gn5 = addBatchNorm2d(network, weights, mask_conv5.get_Output(0), 'mask_feat_head.convs_all_levels.3.conv1.gn', 1e-5)
    mask_relu5 = network.add_activation(input=mask_gn5.get_output(0), type=trt.ActivationType.RELU)

    mask_deconv4_w = weights['mask_feat_head.convs_all_levels.3.upsample1.weight'].numpy()
    mask_deconv4 = network.add_deconvolution(input=mask_relu5.get_output(0), num_output_maps=128, kernel_shape=(2, 2),
                                             kernel=mask_deconv4_w)
    mask_deconv4.stride = (2, 2)
    mask_deconv4.groups = 128

    mask_conv6_w = weights['mask_feat_head.convs_all_levels.3.conv2.conv.weight'].numpy()
    mask_conv6 = network.add_convolution(input=mask_deconv4.get_output(0), num_output_maps=128, kernel_shape=(3, 3),
                                         kernel=mask_conv6_w)
    mask_conv6.stride = (1, 1)
    mask_conv6.padding = (1, 1)
    mask_gn6 = addBatchNorm2d(network, weights, mask_conv6.get_Output(0), 'mask_feat_head.convs_all_levels.3.conv2.gn', 1e-5)
    mask_relu6 = network.add_activation(input=mask_gn6.get_output(0), type=trt.ActivationType.RELU)

    mask_deconv5_w = weights['mask_feat_head.convs_all_levels.3.upsample2.weight'].numpy()
    mask_deconv5 = network.add_deconvolution(input=mask_relu6.get_output(0), num_output_maps=128, kernel_shape=(2, 2),
                                             kernel=mask_deconv5_w)
    mask_deconv5.stride = (2, 2)
    mask_deconv5.groups = 128

    # mask_feat_head.conv_pred
    mask_conv7_w = weights['mask_feat_head.conv_pred.0.conv.weight'].numpy()
    mask_conv7 = network.add_convolution(input=mask_deconv5.get_output(0), num_output_maps=256, kernel_shape=(3, 3),
                                         kernel=mask_conv7_w)
    mask_conv7.stride = (1, 1)
    mask_conv7.padding = (1, 1)
    mask_gn7 = addBatchNorm2d(network, weights, mask_conv7.get_Output(0), 'mask_feat_head.convs_pred.0.gn', 1e-5)
    mask_relu7 = network.add_activation(input=mask_gn7.get_output(0), type=trt.ActivationType.RELU)


    # bbox_head
    # cate_convs
    bbox_conv0_w = weights['bbox_head.cate_convs.0.conv.weight'].numpy()
    bbox_conv0 = network.add_convolution(input=mask_relu7.get_output(0), num_output_maps=512, kernel_shape=(3, 3),
                                         kernel=bbox_conv0_w)
    bbox_conv0.stride = (1, 1)
    bbox_conv0.padding = (1, 1)
    bbox_gn0 = addBatchNorm2d(network, weights, bbox_conv0.get_Output(0), 'bbox_head.cate_convs.0.gn', 1e-5)
    bbox_relu0 = network.add_activation(input=bbox_gn0.get_output(0), type=trt.ActivationType.RELU)

    bbox_conv1_w = weights['bbox_head.cate_convs.1.conv.weight'].numpy()
    bbox_conv1 = network.add_convolution(input=bbox_relu0.get_output(0), num_output_maps=512, kernel_shape=(3, 3),
                                         kernel=bbox_conv1_w)
    bbox_conv1.stride = (1, 1)
    bbox_conv1.padding = (1, 1)
    bbox_gn1 = addBatchNorm2d(network, weights, bbox_conv1.get_Output(0), 'bbox_head.cate_convs.1.gn', 1e-5)
    bbox_relu1 = network.add_activation(input=bbox_gn1.get_output(0), type=trt.ActivationType.RELU)

    bbox_conv2_w = weights['bbox_head.cate_convs.1.conv.weight'].numpy()
    bbox_conv2 = network.add_convolution(input=bbox_relu1.get_output(0), num_output_maps=512, kernel_shape=(3, 3),
                                         kernel=bbox_conv2_w)
    bbox_conv2.stride = (1, 1)
    bbox_conv2.padding = (1, 1)
    bbox_gn2 = addBatchNorm2d(network, weights, bbox_conv2.get_Output(0), 'bbox_head.cate_convs.2.gn', 1e-5)
    bbox_relu2 = network.add_activation(input=bbox_gn2.get_output(0), type=trt.ActivationType.RELU)

    bbox_conv3_w = weights['bbox_head.cate_convs.2.conv.weight'].numpy()
    bbox_conv3 = network.add_convolution(input=bbox_relu2.get_output(0), num_output_maps=512, kernel_shape=(3, 3),
                                         kernel=bbox_conv3_w)
    bbox_conv3.stride = (1, 1)
    bbox_conv3.padding = (1, 1)
    bbox_gn3 = addBatchNorm2d(network, weights, bbox_conv3.get_Output(0), 'bbox_head.cate_convs.3.gn', 1e-5)
    bbox_relu3 = network.add_activation(input=bbox_gn3.get_output(0), type=trt.ActivationType.RELU)

    # kernel_convs
    bbox_conv4_w = weights['bbox_head.kernel_convs.0.conv.weight'].numpy()
    bbox_conv4 = network.add_convolution(input=bbox_relu3.get_output(0), num_output_maps=512, kernel_shape=(3, 3),
                                         kernel=bbox_conv4_w)
    bbox_conv4.stride = (1, 1)
    bbox_conv4.padding = (1, 1)
    bbox_gn4 = addBatchNorm2d(network, weights, bbox_conv4.get_Output(0), 'bbox_head.kernel_convs.0.gn', 1e-5)
    bbox_relu4 = network.add_activation(input=bbox_gn4.get_output(0), type=trt.ActivationType.RELU)

    bbox_conv5_w = weights['bbox_head.cate_convs.1.conv.weight'].numpy()
    bbox_conv5 = network.add_convolution(input=bbox_relu4.get_output(0), num_output_maps=512, kernel_shape=(3, 3),
                                         kernel=bbox_conv5_w)
    bbox_conv5.stride = (1, 1)
    bbox_conv5.padding = (1, 1)
    bbox_gn5 = addBatchNorm2d(network, weights, bbox_conv5.get_Output(0), 'bbox_head.kernel_convs.1.gn', 1e-5)
    bbox_relu5 = network.add_activation(input=bbox_gn5.get_output(0), type=trt.ActivationType.RELU)

    bbox_conv6_w = weights['bbox_head.cate_convs.2.conv.weight'].numpy()
    bbox_conv6 = network.add_convolution(input=bbox_relu5.get_output(0), num_output_maps=512, kernel_shape=(3, 3),
                                         kernel=bbox_conv6_w)
    bbox_conv6.stride = (1, 1)
    bbox_conv6.padding = (1, 1)
    bbox_gn6 = addBatchNorm2d(network, weights, bbox_conv6.get_Output(0), 'bbox_head.kernel_convs.2.gn', 1e-5)
    bbox_relu6 = network.add_activation(input=bbox_gn6.get_output(0), type=trt.ActivationType.RELU)

    bbox_conv7_w = weights['bbox_head.cate_convs.3.conv.weight'].numpy()
    bbox_conv7 = network.add_convolution(input=bbox_relu6.get_output(0), num_output_maps=512, kernel_shape=(3, 3),
                                         kernel=bbox_conv7_w)
    bbox_conv7.stride = (1, 1)
    bbox_conv7.padding = (1, 1)
    bbox_gn7 = addBatchNorm2d(network, weights, bbox_conv7.get_Output(0), 'bbox_head.kernel_convs.3.gn', 1e-5)
    bbox_relu7 = network.add_activation(input=bbox_gn7.get_output(0), type=trt.ActivationType.RELU)

    # solo_cate
    solo_cate_conv_w = weights['bbox_head.cate_convs.weight'].numpy()
    solo_cate_conv_b = weights['bbox_head.cate_convs.bias'].numpy()
    solo_cate_conv = network.add_convolution(input=bbox_relu7.get_output(0), num_output_maps=80, kernel_shape=(3, 3),
                                         kernel=solo_cate_conv_w, bias = solo_cate_conv_b)
    solo_cate_conv.stride = (1, 1)
    solo_cate_conv.padding = (1, 1)


    # solo_kernel

    solo_kernel_conv_w = weights['bbox_head.cate_convs.3.conv.weight'].numpy()
    solo_kernel_conv_b = weights['bbox_head.cate_convs.3.conv.bias'].numpy()
    solo_kernel_conv = network.add_convolution(input=solo_cate_conv.get_output(0), num_output_maps=80, kernel_shape=(3, 3),
                                             kernel=solo_kernel_conv_w, bias=solo_kernel_conv_b)
    solo_kernel_conv.stride = (1, 1)
    solo_kernel_conv.padding = (1, 1)

    return network

def build_engine(weights):

    with trt.Builder(logger) as builder, builder.create_network() as network:
        builder.max_workspace_size = 1<<30
        solov2_network(network, weights)

        # 构建并返回一个engine
        return builder.build_cuda_engine(network)

# 分配buffer
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # 分配host和device端的buffer
        host_men = cuda.pagelocked_empty(size, dtype)
        device_men = cuda.men_alloc(host_men.nbytes)

        # 将device端的buffer追加到device的building
        bindings.append(int(device_men))

        if engine.binding_is_input(binding):
            inputs.append(cuda.mem_alloc(host_men, device_men, stream))
        else:
            outputs.append(cuda.mem_alloc(host_men, device_men, stream))
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    #将数据移动到GPU
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

    #执行inference
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)

    # 将结果从GPU写回到host端
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

    # 同步stream
    stream.synchronize()

    # 返回host端的输出结果
    return [out.host for out in outputs]


def main():
    w = torch.load('/home/sqc/learn/code/segmentation/SOLO/checkpoints/SOLOv2_R101_3x.pth', map_location='cpu')
    w1 = []
    for k in zip(w['state_dict'].keys(), w['state_dict'].values()):
        w1.append(k)
    weights = dict(w1)
    print('len:',len( weights ))

    # 基于build_engine构建engine; 用tensorrt来进行inference
    with build_engine(weights) as engine:
        # 构建engine，分配buffer，创建一个流
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        with engine.create_execution_context() as  context:
            # 读取样本
            img = cv2.imread("image.jpg")

            # 进行inference
            [output] = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)


if __name__=='__main__':
    main()
