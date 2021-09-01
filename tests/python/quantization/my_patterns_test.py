import os
os.environ['DNNL_VERBOSE'] = '2'
import mxnet as mx
from mxnet.gluon import nn
from mxnet.io import NDArrayIter
from mxnet.test_utils import DummyIter, environment, assert_almost_equal_with_err
from common import with_seed


SHAPES = [(1, 224)] #, (16, 1024)]
num_hidden = [512] #, 1024]
SHAPES4D = [(1, 1, 64, 64)] #, (1, 16, 224, 224)]
rounds = 2

class FCFC(nn.HybridBlock):
    def __init__(self, num_in, num_hidden, **kwargs):
        super(FCFC, self).__init__(**kwargs)
        self.fc0 = nn.Dense(units=num_hidden, in_units=num_in)
        self.fc1 = nn.Dense(units=num_hidden)
    
    def hybrid_forward(self, F, x):
        out = self.fc1(self.fc0(x))
        return out

class ConvConv(nn.HybridBlock):
    def __init__(self, use_bias, **kwargs):
        super(ConvConv, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias)
        self.conv1 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias)
    
    def hybrid_forward(self, F, x):
        out = self.conv1(self.conv0(x))
        return out

class CalibIter(mx.io.DataIter):
    def __init__(self, batch, data_shape, batch_size):
        super(CalibIter, self).__init__(batch_size)
        self.label_shape = (batch_size,)
        self.data_shape = data_shape
        if isinstance(data_shape, tuple):
          self.provide_data = [('data', data_shape)]
        else:
          self.provide_data = data_shape
        self.provide_label = []
        self.batch = batch

    def __iter__(self):
        yield self.batch


def test1():
    print("Checking fc->fc and quant->fc pattern")
    for shape in SHAPES:
        for nhid in num_hidden:
            net = FCFC(shape[1], nhid)
            net.initialize()
            net.hybridize(static_alloc=True, static_shape=True)
            x = mx.nd.random_uniform(shape=shape, low=-1.0, high=1.0)
            batch = mx.io.DataBatch([x], [])
            calib_data = CalibIter(batch, [mx.io.DataDesc("data", shape=shape, dtype='float32')], 1)
            net_quantized = mx.contrib.quant.quantize_net_v2(net, quantized_dtype='auto',
                                                                exclude_layers=None,
                                                                exclude_layers_match=None,
                                                                calib_data=calib_data,
                                                                calib_mode='naive',
                                                                num_calib_examples=1,
                                                                ctx=mx.current_context())
            for i in range(rounds):
                print("Round ", i)
                o = net_quantized(x)
                o.wait_to_read()
            print(f"Shape: {shape}")

def test2():
    print("Checking conv->conv pattern")
    for shape in SHAPES4D:
        net = ConvConv(1)
        net.initialize()
        net.hybridize(static_alloc=True, static_shape=True)
        x = mx.nd.random_uniform(shape=shape, low=-1.0, high=1.0)
        batch = mx.io.DataBatch([x], [])
        calib_data = CalibIter(batch, [mx.io.DataDesc("data", shape=shape, dtype='float32')], 1)
        net_quantized = mx.contrib.quant.quantize_net_v2(net, quantized_dtype='auto',
                                                            exclude_layers=None,
                                                            exclude_layers_match=None,
                                                            calib_data=calib_data,
                                                            calib_mode='naive',
                                                            num_calib_examples=1,
                                                            ctx=mx.current_context())
        net_quantized.hybridize(static_alloc=True, static_shape=True)
        for i in range(rounds):
            print("Round ", i)
            o = net_quantized(x)
            o.wait_to_read()
        print(f"Shape: {shape}")



def is_test_for_mkldnn():
    return (mx.current_context().device_type == 'cpu'
            and os.environ.get('ENABLE_MKLDNN_QUANTIZATION_TEST') == '1')

def test_onednn_shifted_conv_conv():
    mx.random.seed(4)
    batch_size = 2
    if not is_test_for_mkldnn():
        print("Test only for mkldnn")
        return

    def get_conv_conv_layers():
        net = mx.gluon.nn.HybridSequential()
        with net.name_scope():
            net.add(mx.gluon.nn.Conv2D(channels=2, kernel_size=(2, 2),
                                       strides=1, use_bias=True))
            net.add(mx.gluon.nn.Conv2D(channels=2, kernel_size=(2, 2),
                                       strides=1, use_bias=True,))
        net.initialize()
        return net

    def quantize_net(net, qdtype, random_data):
        calib_data = NDArrayIter(data=random_data, batch_size=batch_size)
        calib_data = DummyIter(calib_data)
        net = mx.contrib.quant.quantize_net(net, quantized_dtype=qdtype,
                                            exclude_layers=None,
                                            exclude_layers_match=None,
                                            calib_data=calib_data,
                                            calib_mode='naive',
                                            num_calib_examples=1,
                                            ctx=mx.current_context())
        net.hybridize(static_alloc=True, static_shape=True)
        print("calibrated, now run to get symbol")
        out = net(random_data)
        out.wait_to_read()

        _, sym = net._cached_graph
        conv0_attrs = sym.attr_dict()["quantized_sg_mkldnn_conv_0"]
        conv1_attrs = sym.attr_dict()["quantized_sg_mkldnn_conv_1"]
        return conv0_attrs, conv1_attrs, out

    def check(qdtype, random_data):
        net_ref = get_conv_conv_layers()
        out_ref = net_ref(random_data)
        out_ref.wait_to_read()
        wagi = net_ref.collect_params()['hybridsequential0_conv0_weight'].data()
        net_ref.collect_params()['hybridsequential0_conv0_weight'].data()[:] = 1 # mx.nd.arange(wagi.size).reshape_like(wagi) - wagi.size//2
        wagi2 = net_ref.collect_params()['hybridsequential0_conv1_weight'].data()
        net_ref.collect_params()['hybridsequential0_conv1_weight'].data()[:] = 2 # mx.nd.arange(wagi2.size).reshape_like(wagi2) - wagi2.size//2
        conv0_attrs, conv1_attrs, out_q = quantize_net(net_ref, qdtype, random_data)

 

        min_range = mx.nd.min(out_ref).asscalar()
        max_range = mx.nd.max(out_ref).asscalar()
        atol = 0.001 * max(abs(min_range), abs(max_range))
        assert_almost_equal_with_err(out_q, out_ref, rtol=0.001, atol=atol, etol=0.002)

        if qdtype == 'auto':
            print("checking if shifted")
            assert conv0_attrs['shifted_output'] == 'True'
            assert conv1_attrs['shifted_input'] == 'True'

    with environment({'MXNET_DISABLE_SHIFTED_QUANTIZATION_OPTIMIZATIONS': '0',
                        'MXNET_DISABLE_SHIFTED_CONV_CONV_OPTIMIZATION' : '0',
                        'ENABLE_MKLDNN_QUANTIZATION_TEST' : '1' }):

        for qdtype in ['auto']:
            new_data = mx.nd.random_uniform(low=0 if qdtype == 'uint8' else -1, high=1, shape=(batch_size, 1, 4, 4))
            print(new_data)
            check(qdtype, new_data)

#test2()
test_onednn_shifted_conv_conv()