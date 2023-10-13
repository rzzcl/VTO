import os
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from dataset import *
from model_feature_unet import UNet
from networks import ConditionGenerator, VGGLoss
import argparse
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from cloth_dataset import kaggle2016nerve
from PIL import Image
import torchvision.utils as vutils
from torch.autograd import Variable
import shutil
from cp_dataset import CPDataset, CPDataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot',type=str, default='/data1/diffusion_virtual_try_on/dataset/train/cloth/', help='path to dataset of kaggle ultrasound nerve segmentation')
# parser.add_argument('dataroot', default='data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--start_epoch', type=int, default=0, help='number of epoch to start')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda'  , default='True', help='enables cuda')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--useBN', action='store_true', help='enalbes batch normalization')
parser.add_argument('--output_name', default='checkpoint___.tar', type=str, help='output checkpoint filename')

args = parser.parse_args()
print(args)

############## dataset processing
# train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                               transforms.CenterCrop(600),
#                                               transforms.Resize(256)])
train_transforms = transforms.Compose([transforms.Resize((256,192)),
                                      transforms.ToTensor(),   
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = kaggle2016nerve(args.dataroot,train_transforms)

# train_dataset = CPDataset(opt)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                           num_workers=args.workers, shuffle=True)



# create dataloader
# train_loader = CPDataLoader(opt, train_dataset)

############## create model
# model = UNet(args.useBN)
model = UNet()
if args.cuda:
  model.cuda()
  cudnn.benchmark = True

############## resume
if args.resume:
  if os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))

    if args.cuda == False:
      checkpoint = torch.load(args.resume, map_location={'cuda:0':'cpu'})

    args.start_epoch = checkpoint['epoch']

    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint (epoch {}, loss {})"
        .format(checkpoint['epoch'], checkpoint['loss']) )
  else:
    print("=> no checkpoint found at '{}'".format(args.resume))


def save_checkpoint(state, filename=args.output_name):
  torch.save(state, filename)

############## training
optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
model.train()

def train(args):
  """
  training
  """
  for epoch in range(args.epochs):
    loss_fn = nn.MSELoss()  #L2损失
    criterionVGG = VGGLoss(args)  #感知损失
    if args.cuda:
      loss_fn = loss_fn.cuda()

    loss_sum = 0

    for i, (x,y_true) in tqdm(enumerate(train_loader),total = len(train_loader)):
      # x, y_true = Variable(x), Variable(y)
      if args.cuda:
        x = x.cuda()
        y_true = y_true.cuda()
        print(x.shape,y_true.shape)
        # x = inputs['cloth']['paired'].cuda()
        # x = F.interpolate(x, size=(256, 192), mode='bilinear')
        # y_true = F.interpolate(y_true, size=(2  6, 192), mode='bilinear')
      

      for ii in range(1):
        y_pred,_ = model(x)

        lossL2 = loss_fn(y_pred, y_true)   #L2损失
        loss_vgg = criterionVGG(y_pred, y_true)  #感知损失
        loss = lossL2 + loss_vgg

        optimizer.zero_grad()
        loss.backward()
        loss_sum += loss.item()

        optimizer.step()
  
      if i % 100 == 0:
        print('batch no.{}, loss: {}'.format(i, loss.item()))
        y_pred = y_pred.cpu().detach()
        y_true = y_true.cpu().detach()
        vutils.save_image(y_true,
                              os.path.join('/data1/diffusion_virtual_try_on/result_cloth/',str(epoch)+'_'+str(i)+'_GT.png'),
                              normalize=True,
                              format='png',
                              nrow=4)
        vutils.save_image(y_pred,
                              os.path.join('/data1/diffusion_virtual_try_on/result_cloth/',str(epoch)+'_'+str(i)+'.png'),
                              normalize=True,
                              format='png',
                              nrow=4)
    print('epoch: {}, epoch loss: {}'.format(epoch,loss.item()/len(train_loader) ))

    save_checkpoint({
      'epoch': epoch + 1,
      'state_dict': model.state_dict(),
      'loss': loss.item()/len(train_loader)
    })


train(args)


############ just check test (visualization)

# def showImg(img, binary=True, fName=''):
#   """
#   show image from given numpy image
#   """
#   img = img[0,0,:,:]

#   if binary:
#     img = img > 0.5

#   img = Image.fromarray(np.uint8(img*255), mode='L')

#   if fName:
#     img.save('assets/'+fName+'.png')
#   else:
#     img.show()


# model.eval()
# train_loader.batch_size=1

# for i, (x,y) in enumerate(train_loader):
#   if i >= 11:
#     break
#   print("@@@@@@@@")
#   y_pred = model(Variable(x))
#   showImg(x.numpy(), binary=False, fName='ori_'+str(i))
#   showImg(y_pred.data.numpy(), binary=False, fName='pred_'+str(i))
#   showImg(y.numpy(), fName='gt_'+str(i))
