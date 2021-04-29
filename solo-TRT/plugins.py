from functools import reduce
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import ctypes
np.set_printoptions(threshold=np.inf)

ctypes.cdll.LoadLibrary('/home/sqc/learn/code/segmentation/SOLO/libgroup_normal_plugin.so')
def get_plugin_creator(plugin_name):
    plugin_creator_list = trt.get_plugin_creator().plugin_creator_list
    plugin_creator = None
    for c in plugin_creator_list:
        if c.name == plugin_name:
            plugin_creator=c
    return plugin_creator

def create_gn_plugin(weights, lname):
    '''
    lname=bbox_head.kernel_convs.0.
    '''
    plugin_creator = get_plugin_creator('GroupNormalizationPlugin')
    if plugin_creator == None:
        print('GroupNormalizationPlugin not found. Exiting')
        exit()

    pfc=trt.PluginFieldCollection(
        [trt.PluginField('eps', np.array([1e-05], dtype=np.float32),     trt.PluginFieldType.FLOAT32),
            trt.PluginField('num_groups', np.array([32],dtype=np.int32), trt.PluginFieldType.INT32),
            trt.PluginField('batchSize', np.array([1],  dtype=np.int32), trt.PluginFieldType.INT32), 
            trt.PluginField('channels', np.array([512], dtype=np.int32), trt.PluginFieldType.INT32),
        ])
    gn_weight = weights[lname+'.gn.weight'].numpy()
    gn_bias = weights[lname+'.gn.bias'].numpy()
    pfc.append( trt.PluginField('scale', gn_weight, trt.PluginFieldType.FLOAT32) )
    pfc.append( trt.PluginField('bias', gn_bias, trt.PluginFieldType.FLOAT32))

    gp_plugin = plugin_creator.create_plugin('GroupNormalizationPlugin', pfc)
    return gp_plugin

def conv_gn_relu(network, weights, input_layer, kernel_size ,out_ch, lname):
    '''
    lname=bbox_head.kernel_convs.0.
    '''
    conv_weights = weights[lname+'conv.weight'].numpy()
    kernel_conv = network.add_convolution(input=input_layer.get_output(0), \
        num_output_maps=out_ch, kernel_shape=kernel_size, kernel=conv_weights)
    kernel_conv.strides =(1,1)
    kernel_conv.padding =(1,1)
    kernel_conv.kernel_size=(3,3)

    gn_plugin = create_gn_plugin(weights, lname)
    gn_layer = network.add_plugin_v2([kernel_conv.get_output(0)], gn_plugin)
    relu_layer = network.add_activation(gn_layer.get_output(0), trt.ActivationType.RELU)
    return relu_layer