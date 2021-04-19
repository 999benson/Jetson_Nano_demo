import numpy as np
import cv2
import os
from typing import Tuple
import io
import tvm
import tvm.relay
import time
import onnx
import torch
import torchvision
import torch.onnx
from PIL import Image
import tvm.contrib.graph_runtime as graph_runtime
from mobilenet_v2_tsm import MobileNetV2


def torch2tvm_module(torch_module: torch.nn.Module, torch_inputs: Tuple[torch.Tensor, ...], target):
    torch_module.eval()
    input_names = []
    input_shapes = {}
    with torch.no_grad():
        for index, torch_input in enumerate(torch_inputs):
            name = "i" + str(index)
            input_names.append(name)
            input_shapes[name] = torch_input.shape
        buffer = io.BytesIO()
        torch.onnx.export(torch_module, torch_inputs, buffer, input_names=input_names, output_names=[
                          "o" + str(i) for i in range(len(torch_inputs))], opset_version=10)
        outs = torch_module(*torch_inputs)
        buffer.seek(0, 0)
        onnx_model = onnx.load_model(buffer)
        from onnxsim import simplify
        # this simplifier removes conversion bugs.
        onnx_model, success = simplify(onnx_model)
        assert success
        relay_module, params = tvm.relay.frontend.from_onnx(
            onnx_model, shape=input_shapes)
    with tvm.relay.build_config(opt_level=3):
        graph, tvm_module, params = tvm.relay.build(
            relay_module, target, params=params)
    return graph, tvm_module, params


def torch2executor(torch_module: torch.nn.Module, torch_inputs: Tuple[torch.Tensor, ...], target):
    prefix = f"mobilenet_tsm_tvm_{target}"
    lib_fname = f'{prefix}.tar'
    graph_fname = f'{prefix}.json'
    params_fname = f'{prefix}.params'
    if os.path.exists(lib_fname) and os.path.exists(graph_fname) and os.path.exists(params_fname):
        with open(graph_fname, 'rt') as f:
            graph = f.read()
        tvm_module = tvm.module.load(lib_fname)
        params = tvm.relay.load_param_dict(
            bytearray(open(params_fname, 'rb').read()))
    else:
        graph, tvm_module, params = torch2tvm_module(
            torch_module, torch_inputs, target)
        tvm_module.export_library(lib_fname)
        with open(graph_fname, 'wt') as f:
            f.write(graph)
        with open(params_fname, 'wb') as f:
            f.write(tvm.relay.save_param_dict(params))

    ctx = tvm.gpu() if target.startswith('cuda') else tvm.cpu()
    graph_module = graph_runtime.create(graph, tvm_module, ctx)
    for pname, pvalue in params.items():
        graph_module.set_input(pname, pvalue)

    def executor(inputs: Tuple[tvm.nd.NDArray]):
        for index, value in enumerate(inputs):
            graph_module.set_input(index, value)
        graph_module.run()
        return tuple(graph_module.get_output(index) for index in range(len(inputs)))

    return executor, ctx


def get_executor(use_gpu=True):
    torch_module = MobileNetV2(n_class=27)
    # checkpoint not downloaded
    if not os.path.exists("mobilenetv2_jester_online.pth.tar"):
        print('Downloading PyTorch checkpoint...')
        import urllib.request
        url = 'https://file.lzhu.me/projects/tsm/models/mobilenetv2_jester_online.pth.tar'
        urllib.request.urlretrieve(url, './mobilenetv2_jester_online.pth.tar')
    torch_module.load_state_dict(torch.load(
        "mobilenetv2_jester_online.pth.tar"))
    torch_inputs = (torch.rand(1, 3, 224, 224),
                    torch.zeros([1, 3, 56, 56]),
                    torch.zeros([1, 4, 28, 28]),
                    torch.zeros([1, 4, 28, 28]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 12, 14, 14]),
                    torch.zeros([1, 12, 14, 14]),
                    torch.zeros([1, 20, 7, 7]),
                    torch.zeros([1, 20, 7, 7]))
    if use_gpu:
        target = 'cuda'
    else:
        target = 'llvm -mcpu=cortex-a72 -target=armv7l-linux-gnueabihf'
    return torch2executor(torch_module, torch_inputs, target)


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Scale(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


def get_transform():
    cropping = torchvision.transforms.Compose([
        GroupScale(256),
        GroupCenterCrop(224),
    ])
    transform = torchvision.transforms.Compose([
        cropping,
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform


catigories = [
    "No gesture",  # 0
    "No gesture",  # 1
    "No gesture",  # 2
    "No gesture",  # 3
    "No gesture",  # 4
    "No gesture",  # 5
    "No gesture",  # 6
    "No gesture",  # 7
    "No gesture",  # 8
    "Shaking Hand",  # 9
    "No gesture",  # 10
    "No gesture",  # 11
    "No gesture",  # 12
    "No gesture",  # 13
    "Stop Sign",  # 14
    "down",  # 15
    "left",  # 16
    "right",  # 17
    "up",  # 18
    "No gesture",  # 19
    "No gesture",  # 20
    "No gesture",  # 21
    "No gesture",  # 22
    "No gesture",  # 23
    "No gesture",  # 24
    "No gesture",  # 25
    "No gesture"  # 26
]


def process_output(idx_, history):
    # idx_: the output of current frame
    # history: a list containing the history of predictions

    max_hist_len = 20  # max history buffer

    # mask out illegal action
    if idx_ in [7, 8, 21, 22, 3]:
        idx_ = history[-1]

    # use only single no action class
    if idx_ == 0:
        idx_ = 2

    # history smoothing
    if idx_ != history[-1]:
        # and history[-2] == history[-3]):
        if not (history[-1] == history[-2]):
            idx_ = history[-1]

    history.append(idx_)
    history = history[-max_hist_len:]

    return history[-1], history


WINDOW_NAME = 'Video Gesture Recognition'


def main():
    print("Open camera...")
    cap = cv2.VideoCapture(0)

    print(cap)

    # set a lower resolution for speed up
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # env variables
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    print("Build transformer...")
    transform = get_transform()
    print("Build Executor...")
    executor, ctx = get_executor()
    buffer = (
        tvm.nd.empty((1, 3, 56, 56), ctx=ctx),
        tvm.nd.empty((1, 4, 28, 28), ctx=ctx),
        tvm.nd.empty((1, 4, 28, 28), ctx=ctx),
        tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
        tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
        tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
        tvm.nd.empty((1, 12, 14, 14), ctx=ctx),
        tvm.nd.empty((1, 12, 14, 14), ctx=ctx),
        tvm.nd.empty((1, 20, 7, 7), ctx=ctx),
        tvm.nd.empty((1, 20, 7, 7), ctx=ctx)
    )
    idx = 0
    history = [2]
    history_logit = []

    i_frame = -1

    print("Ready!")
    while True:
        i_frame += 1
        _, img = cap.read()  # (480, 640, 3) 0 ~ 255
        if i_frame % 2 == 0:  # skip every other frame to obtain a suitable frame rate
            t1 = time.time()
            img_tran = transform([Image.fromarray(img).convert('RGB')])
            input_var = torch.autograd.Variable(
                img_tran.view(1, 3, img_tran.size(1), img_tran.size(2)))
            img_nd = tvm.nd.array(input_var.detach().numpy(), ctx=ctx)
            inputs: Tuple[tvm.nd.NDArray] = (img_nd,) + buffer
            outputs = executor(inputs)
            feat, buffer = outputs[0], outputs[1:]
            assert isinstance(feat, tvm.nd.NDArray)

            idx_ = np.argmax(feat.asnumpy(), axis=1)[0]

            history_logit.append(feat.asnumpy())
            history_logit = history_logit[-12:]
            avg_logit = sum(history_logit)
            idx_ = np.argmax(avg_logit, axis=1)[0]

            idx, history = process_output(idx_, history)

            t2 = time.time()
            print(f"{i_frame} {catigories[idx]}")

            current_time = t2 - t1

        img = cv2.resize(img, (640, 480))
        img = img[:, ::-1]
        height, width, _ = img.shape
        label = np.zeros([height // 10, width, 3]).astype('uint8') + 255

        cv2.putText(label, 'Prediction: ' + catigories[idx],
                    (0, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)
        cv2.putText(label, '{:.1f} Vid/s'.format(1 / current_time),
                    (width - 170, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)

        img = np.concatenate((img, label), axis=0)
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # exit
            break
        
    cap.release()
    cv2.destroyAllWindows()


# main()



class GestureDetector(object):
    def __init__(self):
        print("Build transformer...")
        self.transform = get_transform()
        print("Build Executor...")
        self.executor, self.ctx = get_executor()

        self.idx = 0
        self.i_frame = -1
        self.history = [2]
        self.history_logit = []
        self.buffer = (
            tvm.nd.empty((1, 3, 56, 56), ctx=self.ctx),
            tvm.nd.empty((1, 4, 28, 28), ctx=self.ctx),
            tvm.nd.empty((1, 4, 28, 28), ctx=self.ctx),
            tvm.nd.empty((1, 8, 14, 14), ctx=self.ctx),
            tvm.nd.empty((1, 8, 14, 14), ctx=self.ctx),
            tvm.nd.empty((1, 8, 14, 14), ctx=self.ctx),
            tvm.nd.empty((1, 12, 14, 14), ctx=self.ctx),
            tvm.nd.empty((1, 12, 14, 14), ctx=self.ctx),
            tvm.nd.empty((1, 20, 7, 7), ctx=self.ctx),
            tvm.nd.empty((1, 20, 7, 7), ctx=self.ctx)
        )
        self.current_time = time.time()

    def __call__(self, img):
        self.i_frame += 1

        ret = None
        if self.i_frame % 2 == 0:  # skip every other frame to obtain a suitable frame rate
            t1 = time.time()
            img_tran = self.transform([Image.fromarray(img).convert('RGB')])
            input_var = torch.autograd.Variable(
                img_tran.view(1, 3, img_tran.size(1), img_tran.size(2)))
            img_nd = tvm.nd.array(input_var.detach().numpy(), ctx=self.ctx)
            inputs: Tuple[tvm.nd.NDArray] = (img_nd,) + self.buffer
            outputs = self.executor(inputs)
            feat, self.buffer = outputs[0], outputs[1:]

            assert isinstance(feat, tvm.nd.NDArray)

            idx_ = np.argmax(feat.asnumpy(), axis=1)[0]
            self.history_logit.append(feat.asnumpy())
            self.history_logit = self.history_logit[-12:]
            avg_logit = sum(self.history_logit)
            idx_ = np.argmax(avg_logit, axis=1)[0]

            self.idx, self.history = process_output(idx_, self.history)
            if catigories[self.idx] in [9, 15, 16, 17, 18]:
                ret = catigories[self.idx]  

            t2 = time.time()
            # print(f"{self.i_frame} {catigories[self.idx]}")

            self.current_time = t2 - t1

        img = cv2.resize(img, (640, 480))
        img = img[:, ::-1]
        height, width, _ = img.shape
        label = np.zeros([height // 10, width, 3]).astype('uint8') + 255

        cv2.putText(label, 'Prediction: ' + catigories[self.idx],
                    (0, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)
        cv2.putText(label, '{:.1f} Vid/s'.format(1 / self.current_time),
                    (width - 170, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)

        img = np.concatenate((img, label), axis=0)

        return ret, img

