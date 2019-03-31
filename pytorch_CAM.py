# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

import io
import requests
from PIL import Image
import  torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb
import json
import models.cifar as mymodels

# load original saved file with DataParallel(remove 'module')
from collections import OrderedDict

def load_weight_gpu(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    return new_state_dict

# input image
LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
# LABELS_URL = '/media/jaden/DeepLearningCode/pytorch-classification/data/cifar-100-labels.json'

# IMG_URL = '/media/jaden/DeepLearningCode/object_detection/Mask-RCNN/data/images/img5.jpg'
IMG_URL = '/media/jaden/DeepLearningCode/data/VOC/benchmark_RELEASE/dataset/img/2011_001650.jpg'

# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
model_id = 1
if model_id == 1:
    model_name = 'squeezenet1_1'
    net = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
    model_name = 'resnet18'
    net = models.resnet18(pretrained=True)
    finalconv_name = 'layer4'
elif model_id == 3:
    model_name = 'densenet161'
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'
elif model_id == 0:
    model_name = 'resnet-110'
    net = mymodels.__dict__['resnet'](num_classes=100, depth=110)
    model_file = '/media/jaden/DeepLearningCode/pytorch-classification/checkpoints/cifar100/resnet-110/model_best.pth.tar'
    checkpoint = torch.load(model_file)
    state_dict = load_weight_gpu(checkpoint['state_dict'])
    net.load_state_dict(state_dict)
    finalconv_name = 'layer3'


net.eval()

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

'''transforms.ToTensor()
Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
'''
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

# load image from url
# response = requests.get(IMG_URL)
# img_pil = Image.open(io.BytesIO(response.content))

img_pil = Image.open(IMG_URL)
img_pil.save('test.jpg')

img_tensor = preprocess(img_pil)
print('img_tensor shape ', img_tensor.size())
img_variable = Variable(img_tensor.unsqueeze(0))
logit = net(img_variable)

# download the imagenet category list
classes = {int(key):value for (key, value)
          in requests.get(LABELS_URL).json().items()}

# with open(LABELS_URL) as f:
#   classes = json.load(f)

h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()

# output the prediction
for i in range(0, 10):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

# generate class activation mapping for the top1 prediction
cat_idx = 0 # car
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[cat_idx]])

# render the CAM and output
print('output CAM.jpg for the top1 prediction: %s'%classes[idx[cat_idx]])
img = cv2.imread('test.jpg')
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('CAM-{}.jpg'.format(model_name), result)

