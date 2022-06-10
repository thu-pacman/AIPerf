# AIPerf 二次开发

# 1. 概览

为了使AIPerf支持更新的的计算框架, 需要支持将AutoML搜索出的网络结构转换成计算框架内可训练的模型

我们提供两种路线, 分别是路线A和路线B

![](imgs/migrate.png)


# 2. 路线A

路线A从直接的AutoML结构表示转化为可训练模型, 需要实现三个组件

## 2.1 模型转换函数

`AIPerf/src/sdk/pynni/nni/networkmorphism_tuner/layers.py` 内, 实现一个将抽象网络层翻译为实际层的函数, 可参考:
```python
def to_real_keras_layer(layer):
    """
    Real keras layer.
    """
    from keras import layers
    if is_layer(layer, "Dense"):
        return layers.Dense(layer.units, input_shape=(layer.input_units,))
    if is_layer(layer, "Conv"):
        return layers.Conv2D(
            layer.filters,
            layer.kernel_size,
            input_shape=layer.input.shape,
            strides=layer.stride,
            padding="same",
        )  # padding
    if is_layer(layer, "Pooling"):
        if layer.stride!=None:
            return layers.MaxPool2D(pool_size=(layer.kernel_size, layer.kernel_size), strides=layer.stride, padding='same', data_format=None)
        else:
            return layers.MaxPool2D(2)
    if is_layer(layer, "BatchNormalization"):
        return layers.BatchNormalization(input_shape=layer.input.shape)
    if is_layer(layer, "Concatenate"):
        return layers.Concatenate()
    if is_layer(layer, "Add"):
        return layers.Add()
    if is_layer(layer, "Dropout"):
        return keras_dropout(layer, layer.rate)
    if is_layer(layer, "ReLU"):
        return layers.Activation("relu")
    if is_layer(layer, "Softmax"):
        return layers.Activation("softmax")
    if is_layer(layer, "Flatten"):
        return layers.Flatten()
    if is_layer(layer, "GlobalAveragePooling"):
        return layers.GlobalAveragePooling2D()
    return None  # note: this is not written by original author, feel free to modify if you think it's incorrect

```

## 2.2 
`AIPerf/src/sdk/pynni/nni/networkmorphism_tuner/graph.py` 内实现一个class, 将翻译后layer按照拓扑序拼成真正的model, 可以参考`class TfModel`

```python
class TfModel:
    def __init__(self, graph):
        import tensorflow
        from tensorflow.python.keras import backend as K
        import h5py
        self.graph = graph
        self.layers = []
        for layer in graph.layer_list:
            self.layers.append(to_real_tf_layer(layer))

        # Construct the keras graph.
        # Input
        topo_node_list = self.graph.topological_order
        output_id = topo_node_list[-1]
        input_id = topo_node_list[0]
        input_tensor = tensorflow.keras.layers.Input(
            shape=graph.node_list[input_id].shape)

        node_list = deepcopy(self.graph.node_list)
        node_list[input_id] = input_tensor

        # Output
        for v in topo_node_list:
            for u, layer_id in self.graph.reverse_adj_list[v]:
                layer = self.graph.layer_list[layer_id]
                tf_layer = self.layers[layer_id]

                if isinstance(layer, (StubAdd, StubConcatenate)):
                    edge_input_tensor = list(
                        map(
                            lambda x: node_list[x],
                            self.graph.layer_id_to_input_node_ids[layer_id],
                        )
                    )
                else:
                    edge_input_tensor = node_list[u]

                temp_tensor = tf_layer(edge_input_tensor)
                node_list[v] = temp_tensor

        output_tensor = node_list[output_id]
        output_tensor = tensorflow.keras.layers.Activation("softmax", name="activation_add")(
            output_tensor
        )
        self.model = tensorflow.keras.models.Model(
            inputs=input_tensor, outputs=output_tensor)

        self.count = 0
        self.loadh5 = 0

    def legacy_weights(self):
        return self.layers[self.count].trainable_weights + self.layers[self.count].non_trainable_weights

    def load_attributes_from_hdf5_group(self, group, name):
        if name in group.attrs:
            data = [n.decode('utf8') for n in group.attrs[name]]
        else:
            data = []
            chunk_id = 0
            while '%s%d' % (name, chunk_id) in group.attrs:
                data.extend(
                    [n.decode('utf8') for n in group.attrs['%s%d' % (name, chunk_id)]])
                chunk_id += 1
        return data
```

## 2.3 

使用2.2实现的组件, 实现一个训练脚本, 建议按照`examples/trials/network_morphism/imagenet/imagenet_train.py`, `examples/trials/network_morphism/imagenetTorch/imagenet_train.py`的实现, 首先完成一个`demo.py`使用`resnet50.json`生成一个模型进行训练, 再用具体的训练逻辑替换已有的`imagenet_train.py`的对应部分, 并按照指定格式打印日志

```python
        print("[{}] PRINT Epoch {}/{}".format(
            time.strftime('%Y/%m/%d, %I:%M:%S %p'),
            epoch,
            run_epochs
        ))    
        
        # ...
        # 训练网络

        # 训练完每个epoch后打印对应的指标, 注意时间戳格式
        print('[{}] PRINT - loss: {:.4f}, - accuracy: {:.4f} - val_loss: {:.4f} - val_accuracy: {:.4f}'.format(
                time.strftime('%Y/%m/%d, %I:%M:%S %p'),
                train_loss / (len(train_datasets)), 
                train_acc / (len(train_datasets)),
                eval_loss / (len(val_datasets)),
                eval_acc / (len(val_datasets))
            )
        ) 
```

您可以通过对比 `examples/trials/network_morphism/imagenet/imagenet_train.py` 和 `examples/trials/network_morphism/imagenetTorch/imagenet_train.py` 的不同查看需要二次开发的代码段落

# 3. 路线B


AIPerf目前支持TorchModel, 您可以通过TorchModel生成的ONNXModel 来获得自有框架的模型
```python
class ONNXModel:
    def __init__(self, graph):
        self.torch_model = TorchModel(graph)
        from torch.autograd import Variable
        import torch.onnx
        import torchvision

        dummy_input = Variable(torch.randn(10, 3, 224, 224)).cuda()
        torch.onnx.export(self.torch_model, dummy_input, "MODEL.proto", verbose=True)

```

之后的工作请参考`2.3`