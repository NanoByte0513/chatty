# automatically generated by the FlatBuffers compiler, do not modify

# namespace: chatty_fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Model(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Model()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsModel(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # Model
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Model
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Model
    def ModelType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Model
    def Tokenizer(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # Model
    def TokenizerAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int8Flags, o)
        return 0

    # Model
    def TokenizerLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Model
    def TokenizerIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # Model
    def InputEmbed(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from chatty_fbs.LinearLayer import LinearLayer
            obj = LinearLayer()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Model
    def OutputNorm(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from chatty_fbs.Norm import Norm
            obj = Norm()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Model
    def OutputEmbed(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from chatty_fbs.LinearLayer import LinearLayer
            obj = LinearLayer()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Model
    def Layers(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from chatty_fbs.TransformerLayer import TransformerLayer
            obj = TransformerLayer()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Model
    def LayersLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Model
    def LayersIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        return o == 0

    # Model
    def HeadDim(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Model
    def KvNumHeads(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Model
    def QNumHeads(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def ModelStart(builder):
    builder.StartObject(10)

def Start(builder):
    ModelStart(builder)

def ModelAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    ModelAddName(builder, name)

def ModelAddModelType(builder, modelType):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(modelType), 0)

def AddModelType(builder, modelType):
    ModelAddModelType(builder, modelType)

def ModelAddTokenizer(builder, tokenizer):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(tokenizer), 0)

def AddTokenizer(builder, tokenizer):
    ModelAddTokenizer(builder, tokenizer)

def ModelStartTokenizerVector(builder, numElems):
    return builder.StartVector(1, numElems, 1)

def StartTokenizerVector(builder, numElems):
    return ModelStartTokenizerVector(builder, numElems)

def ModelAddInputEmbed(builder, inputEmbed):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(inputEmbed), 0)

def AddInputEmbed(builder, inputEmbed):
    ModelAddInputEmbed(builder, inputEmbed)

def ModelAddOutputNorm(builder, outputNorm):
    builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(outputNorm), 0)

def AddOutputNorm(builder, outputNorm):
    ModelAddOutputNorm(builder, outputNorm)

def ModelAddOutputEmbed(builder, outputEmbed):
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(outputEmbed), 0)

def AddOutputEmbed(builder, outputEmbed):
    ModelAddOutputEmbed(builder, outputEmbed)

def ModelAddLayers(builder, layers):
    builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(layers), 0)

def AddLayers(builder, layers):
    ModelAddLayers(builder, layers)

def ModelStartLayersVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartLayersVector(builder, numElems):
    return ModelStartLayersVector(builder, numElems)

def ModelAddHeadDim(builder, headDim):
    builder.PrependInt32Slot(7, headDim, 0)

def AddHeadDim(builder, headDim):
    ModelAddHeadDim(builder, headDim)

def ModelAddKvNumHeads(builder, kvNumHeads):
    builder.PrependInt32Slot(8, kvNumHeads, 0)

def AddKvNumHeads(builder, kvNumHeads):
    ModelAddKvNumHeads(builder, kvNumHeads)

def ModelAddQNumHeads(builder, qNumHeads):
    builder.PrependInt32Slot(9, qNumHeads, 0)

def AddQNumHeads(builder, qNumHeads):
    ModelAddQNumHeads(builder, qNumHeads)

def ModelEnd(builder):
    return builder.EndObject()

def End(builder):
    return ModelEnd(builder)

import chatty_fbs.LinearLayer
import chatty_fbs.Norm
import chatty_fbs.TransformerLayer
try:
    from typing import List, Optional
except:
    pass

class ModelT(object):

    # ModelT
    def __init__(self):
        self.name = None  # type: str
        self.modelType = None  # type: str
        self.tokenizer = None  # type: List[int]
        self.inputEmbed = None  # type: Optional[chatty_fbs.LinearLayer.LinearLayerT]
        self.outputNorm = None  # type: Optional[chatty_fbs.Norm.NormT]
        self.outputEmbed = None  # type: Optional[chatty_fbs.LinearLayer.LinearLayerT]
        self.layers = None  # type: List[chatty_fbs.TransformerLayer.TransformerLayerT]
        self.headDim = 0  # type: int
        self.kvNumHeads = 0  # type: int
        self.qNumHeads = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        model = Model()
        model.Init(buf, pos)
        return cls.InitFromObj(model)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos+n)

    @classmethod
    def InitFromObj(cls, model):
        x = ModelT()
        x._UnPack(model)
        return x

    # ModelT
    def _UnPack(self, model):
        if model is None:
            return
        self.name = model.Name()
        self.modelType = model.ModelType()
        if not model.TokenizerIsNone():
            if np is None:
                self.tokenizer = []
                for i in range(model.TokenizerLength()):
                    self.tokenizer.append(model.Tokenizer(i))
            else:
                self.tokenizer = model.TokenizerAsNumpy()
        if model.InputEmbed() is not None:
            self.inputEmbed = chatty_fbs.LinearLayer.LinearLayerT.InitFromObj(model.InputEmbed())
        if model.OutputNorm() is not None:
            self.outputNorm = chatty_fbs.Norm.NormT.InitFromObj(model.OutputNorm())
        if model.OutputEmbed() is not None:
            self.outputEmbed = chatty_fbs.LinearLayer.LinearLayerT.InitFromObj(model.OutputEmbed())
        if not model.LayersIsNone():
            self.layers = []
            for i in range(model.LayersLength()):
                if model.Layers(i) is None:
                    self.layers.append(None)
                else:
                    transformerLayer_ = chatty_fbs.TransformerLayer.TransformerLayerT.InitFromObj(model.Layers(i))
                    self.layers.append(transformerLayer_)
        self.headDim = model.HeadDim()
        self.kvNumHeads = model.KvNumHeads()
        self.qNumHeads = model.QNumHeads()

    # ModelT
    def Pack(self, builder):
        if self.name is not None:
            name = builder.CreateString(self.name)
        if self.modelType is not None:
            modelType = builder.CreateString(self.modelType)
        if self.tokenizer is not None:
            if np is not None and type(self.tokenizer) is np.ndarray:
                tokenizer = builder.CreateNumpyVector(self.tokenizer)
            else:
                ModelStartTokenizerVector(builder, len(self.tokenizer))
                for i in reversed(range(len(self.tokenizer))):
                    builder.PrependByte(self.tokenizer[i])
                tokenizer = builder.EndVector()
        if self.inputEmbed is not None:
            inputEmbed = self.inputEmbed.Pack(builder)
        if self.outputNorm is not None:
            outputNorm = self.outputNorm.Pack(builder)
        if self.outputEmbed is not None:
            outputEmbed = self.outputEmbed.Pack(builder)
        if self.layers is not None:
            layerslist = []
            for i in range(len(self.layers)):
                layerslist.append(self.layers[i].Pack(builder))
            ModelStartLayersVector(builder, len(self.layers))
            for i in reversed(range(len(self.layers))):
                builder.PrependUOffsetTRelative(layerslist[i])
            layers = builder.EndVector()
        ModelStart(builder)
        if self.name is not None:
            ModelAddName(builder, name)
        if self.modelType is not None:
            ModelAddModelType(builder, modelType)
        if self.tokenizer is not None:
            ModelAddTokenizer(builder, tokenizer)
        if self.inputEmbed is not None:
            ModelAddInputEmbed(builder, inputEmbed)
        if self.outputNorm is not None:
            ModelAddOutputNorm(builder, outputNorm)
        if self.outputEmbed is not None:
            ModelAddOutputEmbed(builder, outputEmbed)
        if self.layers is not None:
            ModelAddLayers(builder, layers)
        ModelAddHeadDim(builder, self.headDim)
        ModelAddKvNumHeads(builder, self.kvNumHeads)
        ModelAddQNumHeads(builder, self.qNumHeads)
        model = ModelEnd(builder)
        return model
