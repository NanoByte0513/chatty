import torch

def torch_dtype2dtype(torch_dtype: torch.dtype):
    pass

class ChattyObject():
    def __init__(self, **kwarg):
        for k, v in kwarg.items():
            self.k = v

    def build(self, **kwargs):
        pass

class ChattyScaleInfo(ChattyObject):
    def __init__(self):
        super().__init__()

    def build(self, builder, offset):
        return super().build()


class ChattyTensor(ChattyObject):
    def __init__(self, fake_tensor:dict, scale:dict):
        self.dtype = fake_tensor["dtype"]
        self.shape = fake_tensor["shape"]
        bytes_per_elem = torch.tensor([], dtype=self.dtype).element_size()
        elem_num = self.shape.numel()
        data_size = bytes_per_elem * elem_num
        super().__init__(name=fake_tensor["name"], dtype=self.dtype, shape=self.shape, data_size=data_size, scale=scale)

    def build(self, builder, offset:int, scale_offset:int):
        name = None
        shape = None
        dtype = torch_dtype2dtype(self.dtype)
        scale = ChattyScaleInfo(self.scale).build(builder, scale_offset)
        return super().build(name=name, shape=shape, dtype=dtype, offset=offset, scale=scale)
    

class ChattyNorm(ChattyObject):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)