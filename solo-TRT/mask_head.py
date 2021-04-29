import numpy as np
import tensorrt as trt 
from .plugins import conv_gn_relu

def make_cgr(network, weights, input_layer, idx):
    up_layer = input_layer
    for ridx in range(idx):
        # mask_feat_head.convs_all_levels.2.conv0.
        lname='mask_feat_head.convs_all_levels.'+str(idx)+'.conv'+str(ridx)+'.'
        cgr = conv_gn_relu(network, weights, up_layer, (3,3), 128, lname)
        up_layer = network.add_resize(cgr.get_output(0))
        up_layer.scales=[1, 1, 2, 2]
        up_layer.resize_mode = trt.ResizeMode.LINEAR

    return up_layer

def make_mask_head(network, weights, input_layers):
    cgr_layer=conv_gn_relu(network, weights, input_layers[0], (3,3), out_ch=128, 
        lname='mask_feat_head.convs_all_levels.0.conv0.')

    for i in range(1, len(input_layers)):
        input_layer=input_layers[i]
        if i==3:
            x_range = np.linspace(-1, 1, 38)
            y_range = np.linspace(-1, 1, 25)
            y,x = np.meshgrid(y_range, x_range)
            y.expand([1, 1,-1, -1] )
            x.expand([1, 1,-1, -1] )
            coord_feat = np.concatenate([x,y], 1)
            coord_layer=network.add_constant(shape=(256, 25, 38), weights=coord_feat)
            cat_layer = network.add_concatenation([coord_layer.get_output(0), input_layer.get_output(0)])
            cat_layer.axis=1
            input_layer=cat_layer
        up_layer = make_cgr(network, weights, input_layer, idx=i)
        cgr_layer = network.add_elementwise(cgr_layer.get_output(0), up_layer.get_output(0), trt.ElementWiseOperation.SUM)
    
    feature_pred = conv_gn_relu(network, weights, up_layer,(1,1), 256,
        lname='mask_feat_head.conv_pred.0.')
    return feature_pred

    