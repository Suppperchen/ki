from unet2d import unet_2d
import torch
from PIL import Image
import numpy as np
from torch.autograd import Variable
from matplotlib import pyplot as plt
from loaddata import get_listData, get_new_data, singel_image_preprocessing, get_new_testdata, convert_back, preproImage
import diceloss
import skimage.io
def val(a,b):


    with torch.no_grad():
        data = Variable(torch.Tensor(a))
        data = data.cuda()

        mask = Variable(torch.Tensor(b))
        mask = mask.cuda()

        unet = unet_2d(1, 1)
        unet.cuda()
        unet.load_state_dict(torch.load('test9.pth'))
        out = unet(data)
        print(out.size())
        print(diceloss.DiceLoss()(out,mask))

    np_data = out.cpu().data.numpy()

    plt.figure()
    f, axarr = plt.subplots(3, 1)
    axarr[0].imshow(np_data.reshape(512, 512), cmap='gray')
    axarr[1].imshow(b.reshape(512, 512), cmap='gray')
    axarr[2].imshow(a.reshape(512, 512), cmap='gray')
    plt.show()


def train_show_result():
    #list_train = get_listData('Data/test/image', 'Data/test/mask', 1)
    _, list_label = get_new_data(16)
    for i in range(2,3):
        a,b = list_label[i]
        for i in range(a.shape[0]):

            val(a[i:i+1,:,:,:],b[i:i+1,:,:,:])

def show_singel_result():
    img = np.asarray(Image.open("trainingData/21_training.tiff").resize((512, 512)))
    img = img.astype(float)
    img = img / 255.0

    img = np.reshape(img, (1, 3, 512, 512))

    val(img, img)



def show_image_mask():
    #list_train,list_label = get_new_data(16)
    list_train ,list_label= get_listData('Data/train/image', 'Data/train/mask', 16)
    print(len(list_train))
    print(len(list_label))
    a,b = list_train[44]
    c,d = list_label[18]
    print(a.shape,b.shape,c.shape,d.shape)
    for i in range(10,11):
        a,b = list_train[i]
        for j in range(a.shape[0]):
            plt.figure()
            f, axarr = plt.subplots(2, 1)
            axarr[0].imshow(b[j,:,:,:].reshape(b.shape[2], b.shape[2]), cmap='gray')
            axarr[1].imshow(a[j,:,:,:].reshape(b.shape[2], b.shape[2]), cmap='gray')
            plt.show()


def test_show_result():
    image, mask = get_new_testdata()
    a = image[2]
    b = mask[2]
    print(len(image))
    with torch.no_grad():
        data = Variable(torch.Tensor(a))
        data = data.cuda()

        mask = Variable(torch.Tensor(b))
        mask = mask.cuda()

        unet = unet_2d(1, 1)
        unet.cuda()
        unet.load_state_dict(torch.load('test9.pth'))
        out = unet(data)
        print(out.size())
        print(diceloss.DiceLoss()(out,mask))

    np_data = out.cpu().data.numpy()

    plt.figure()
    f, axarr = plt.subplots(3, 1)
    axarr[0].imshow(convert_back(np_data), cmap='gray')
    axarr[1].imshow(convert_back(b), cmap='gray')
    axarr[2].imshow(convert_back(a), cmap='gray')
    plt.show()

def test():
    eachImgae = skimage.io.imread('Data/test/image/1.png')
    eachImgae = preproImage(eachImgae, 3)

    eachlabel = skimage.io.imread('Data/test/mask/1.png')
    eachlabel = preproImage(eachlabel, 1)

    image_dataset = np.array(eachImgae)
    mask_dataset = np.array(eachlabel)
    a = image_dataset.reshape((image_dataset.shape[0], 1, 64, 64))
    b = mask_dataset.reshape((mask_dataset.shape[0], 1, 64, 64))

    with torch.no_grad():
        data = Variable(torch.Tensor(a))
        data = data.cuda()

        mask = Variable(torch.Tensor(b))
        mask = mask.cuda()

        unet = unet_2d(1, 1)
        unet.cuda()
        unet.load_state_dict(torch.load('test10.pth'))
        out = unet(data)
        print(out.size())
        print(diceloss.DiceLoss()(out,mask))

    np_data = out.cpu().data.numpy()

    plt.figure()
    f, axarr = plt.subplots(3, 1)
    axarr[0].imshow(convert_back(np_data), cmap='gray')
    axarr[1].imshow(convert_back(b), cmap='gray')
    axarr[2].imshow(convert_back(a), cmap='gray')
    plt.show()


show_image_mask()