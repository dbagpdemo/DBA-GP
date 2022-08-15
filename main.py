import torch
import torchvision
import box
import argparse
import torchvision.transforms as T
from torchvision.io import read_image
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--maxq', type=int, default=1000, help='Number of max queries.')
parser.add_argument('--time', action='store_true', default=False, help='Time prior.')
args = parser.parse_args()
#args.time = True


def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

model = torchvision.models.resnet18(pretrained=True).eval()
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
fmodel = box.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
path = './box/data/imagenet_07_609.jpg'
path_tar = './box/data/imagenet_00_243.jpg'


src_image = read_image(path) / 255.0
src_image = src_image.cuda()
tar_image = read_image(path_tar) / 255.0
tar_image = tar_image.cuda()
src_labels = torch.argmax(fmodel(src_image.unsqueeze(0))).unsqueeze(0)
tar_labels = torch.argmax(fmodel(tar_image.unsqueeze(0))).unsqueeze(0)
criterion = box.criteria.TargetedMisclassification(src_labels)
attack = box.attacks.DBA_GP(initial_gradient_eval_steps=100,
                                    max_gradient_eval_steps=100,
                                    steps=200,
                                    args=args,
                                    max_queries=args.maxq,
                                    gamma=1500,
                                )
adv, clipped, success = attack(fmodel, tar_image.unsqueeze(0), criterion,
                                   starting_points=src_image.unsqueeze(0), epsilons=None)
out_image = adv
show([out_image[0]])
plt.show()
