import tensorrt as trt
import numpy as np
import torch
from .plugins import conv_gn_relu

def resize_and_split(network, input_layers):
    layer0 = input_layers[0]
    resize_layer0=network.add_resize(layer0.get_output(0))
    resize_layer0.scales = [1, 1, 0.5, 0.5]
    resize_layer0.resize_mode = trt.ResizeMode.LINEAR
    input_layers[0] = resize_layer0

    layer4 = input_layers[4]
    resize_layer4=network.add_resize(layer4.get_output(0))
    resize_layer4.scales = [1,1,2,2]
    resize_layer0.resize_mode = trt.ResizeMode.LINEAR
    input_layers[4] = resize_layer4
    return input_layers

def points_nms(network, input_layer, kernel):
    heat_layer = network.add_activation(input=input_layer.get_output(0), type=trt.ActivationType.SIGMOID)
    pool_layer = network.add_pooling(heat_layer.get_output(0), type=trt.PoolingType.MAX, window_size=kernel)
    pool_layer.strides=(1,1)
    pool_layer.padding=(1,1)
    pshape = pool_layer.get_output(0).shape
    cut_layer = network.add_slice(pool_layer.get_output(0), start=(0,0,0), shape=(pshape[0], pshape[1]-1, pshape[2]-1) )

    keep_layer = network.add_elementwise(pool_layer.get_output(0), cut_layer.get_output(0), trt.ElementWiseOperation.EQUAL)
    nms_layer = network.add_elementwise(heat_layer.get_output(0), keep_layer.get_output(0).float(), trt.ElementWiseOperation.PROD)
    return nms_layer

def solov2_head(network, weights, input_layers):
    seg_num_grids=[40, 36, 24, 16, 12]
    new_feats = resize_and_split(network, input_layers)
    '''
    add_slice(self: 
   tensorrt.tensorrt.INetworkDefinition,
    input:  tensorrt.tensorrt.ITensor, 
    start:  tensorrt.tensorrt.Dims, 
    shape:  tensorrt.tensorrt.Dims, 
    stride: tensorrt.tensorrt.Dims) → tensorrt.tensorrt.ISliceLayer
    '''
    cate_preds=[]
    solo_preds=[]
    for i,feat in enumerate(new_feats):
        x_range=np.linspace(-1, 1, feat.shape[-1])
        y_range=np.linspace(-1, 1, feat.shape[-2])
        y,x = np.meshgrid(y_range, x_range)
        y = y.expand([feat.shape[0], 1, -1, -1])
        x = x.expand([feat.shape[0], 1, -1, -1])
        coord_feat = np.concatenate([x,y], 1)

        coord_const = network.add_constant(shape=coord_feat.shape, weights=coord_feat)
        kernel_feat= network.add_concatenation([feat.get_output(0), coord_const.get_output(0)], axis=1)
        seg_grid = seg_num_grids[i]
        resize_feat=network.add_resize(kernel_feat.get_output(0))
        resize_feat.shape=[1, 258, seg_grid, seg_grid]
        resize_feat.resize_mode = trt.ResizeMode.LINEAR

        cate_feat = network.add_slice(resize_feat.get_output(0), start=(0,0,0),
            shape=(resize_feat.shape[0]-2, resize_feat.shape[1],resize_feat.shape[2]) ,stride=(1, 1, 1))
        
        lname = 'bbox_head.kernel_convs.'+str(i)+'.'
        kernel_layer = conv_gn_relu(network, weights, resize_feat, (3,3), 512, lname)
        solo_pred_layer = network.add_convolution(kernel_layer.get_output(0), num_output_maps=256,
            kernel_shape=(3,3), kernel=weights['bbox_head.solo_kernel.weight'].numpy(),
            bias=weights['bbox_head.solo_kernel.bias'].numpy())

        lname = 'bbox_head.cate_convs.'+str(i)+'.'
        cate_layer = conv_gn_relu(network, weights, cate_feat, (3,3), 512, lname)
        cate_pred_layer = network.add_convolution(input=cate_layer.get_output(0), num_output_maps=80,
            kernel_shape=(3, 3), kernel=weights['bbox_head.solo_cate.weight'].numpy(),
            bias=weights['bbox_head.solo_cate.bias'].numpy() )
        cate_pred_layer = points_nms(network, cate_pred_layer, kernel=(2,2))
        cate_preds.append(cate_pred_layer)
        solo_preds.append(solo_pred_layer)
    return cate_preds, solo_preds
