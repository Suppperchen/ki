from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
import cv2
import skimage.io
from patchify import patchify


#total 80 training data, 'Data/train/image', 'Data/train/mask'
#total 20 test data, 'Data/test/image','Data/test/mask'
def get_listData(path1, path2, batch):

    list_image = []
    arr = os.listdir(path1)
    for path in arr:
        eachImgae = skimage.io.imread(path1 + '/'+path)
        eachImgae = preproImage_test(eachImgae,3)
        list_image = list_image + eachImgae
    list_label = []
    arr = os.listdir(path2)
    for path in arr:

        eachlabel = skimage.io.imread(path2 + '/'+path)
        eachlabel = preproImage_test(eachlabel, 1)

        list_label = list_label + eachlabel

    #SEED = 448
    #random.seed(SEED)
    #random.shuffle(list_image)
    #random.shuffle(list_label)

    image_dataset = np.array(list_image)
    mask_dataset = np.array(list_label)
    a = image_dataset.reshape((image_dataset.shape[0], 1, 512, 512))
    b = mask_dataset.reshape((mask_dataset.shape[0], 1, 512, 512))




    x_train = a[0:56,:,:,:]
    x_test = a[56:80,:,:,:]
    y_train = b[0:56,:,:,:]
    y_test = b[56:80,:,:,:]



    x_train = batch_numpy(x_train, batch)
    x_test = batch_numpy(x_test, batch)
    y_train = batch_numpy(y_train, batch)
    y_test = batch_numpy(y_test, batch)

    list_1 = zip(x_train, y_train)
    list_2 = zip(x_test, y_test)
    return list(list_1), list(list_2)

#no use
def preproImage(img,channel):
    if channel == 3:
        img = img[:,:,1]
        #img = clahe_equalized(img)

    img = Image.fromarray(img)

    img = np.array(img)
    patches_img = patchify(img, (64, 64),
                               step=64)
    image_dataset =[]
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, :, :]
            single_patch_img = (single_patch_img.astype('float32')) / 255.
            image_dataset.append(single_patch_img)

    return image_dataset


def preproImage_test(img,channel):
    if channel == 3:
        img = img[:,:,1]
        #img = clahe_equalized(img)

    img = Image.fromarray(img)

    img = np.array(img)
    img = (img.astype('float32'))/255

    return [img]





########################try with new data, no use
def clahe_equalized(imgs):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = clahe.apply(imgs)
    return imgs_equalized



def singel_image_preprocessing(image_name,path1,patch_size):
    image_dataset = []
    if image_name.endswith(".jpg") or image_name.endswith(".JPG"):
        image = skimage.io.imread(path1 + "/" + image_name)  # Read image
        image = image[:, :, 1]  # selecting green channel
        image = clahe_equalized(image)  # applying CLAHE
        SIZE_X = (image.shape[1] // patch_size) * patch_size  # getting size multiple of patch size
        SIZE_Y = (image.shape[0] // patch_size) * patch_size  # getting size multiple of patch size
        image = Image.fromarray(image)
        image = image.resize((SIZE_X, SIZE_Y))  # resize image
        image = np.array(image)
        patches_img = patchify(image, (patch_size, patch_size),
                               step=patch_size)  # create patches(patch_sizexpatch_sizex1)

        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i, j, :, :]
                single_patch_img = (single_patch_img.astype('float32')) / 255.
                image_dataset.append(single_patch_img)
        return image_dataset
    if image_name.endswith(".tif"):
        mask = skimage.io.imread(path1 + "/" + image_name)  # Read masks
        SIZE_X = (mask.shape[1] // patch_size) * patch_size  # getting size multiple of patch size
        SIZE_Y = (mask.shape[0] // patch_size) * patch_size  # getting size multiple of patch size
        mask = Image.fromarray(mask)
        mask = mask.resize((SIZE_X, SIZE_Y))  # resize image
        mask = np.array(mask)
        patches_mask = patchify(mask, (patch_size, patch_size),
                                step=patch_size)  # create patches(patch_sizexpatch_sizex1)

        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):
                single_patch_mask = patches_mask[i, j, :, :]
                single_patch_mask = (single_patch_mask.astype('float32')) / 255.
                image_dataset.append(single_patch_mask)
        return image_dataset
#no use
def get_new_data(batch):
    path1 = 'C:/PycharmProjects/Retinal-Vessel-Segmentation-using-variants-of-UNET/images'  # training images directory
    path2 = 'C:/PycharmProjects/Retinal-Vessel-Segmentation-using-variants-of-UNET/manual1'  # training masks directory

    image_dataset = []
    mask_dataset = []

    patch_size = 512

    images = sorted(os.listdir(path1))  # 45 images (2336,3504), SIZE_X  = 2048, SIZE_Y = 3072
    for i, image_name in enumerate(images):
        image_dataset = image_dataset +singel_image_preprocessing(image_name,path1,patch_size)

    masks = sorted(os.listdir(path2))
    for i, mask_name in enumerate(masks):
        mask_dataset = mask_dataset +singel_image_preprocessing(mask_name,path2,patch_size)


    image_dataset = np.array(image_dataset)
    mask_dataset = np.array(mask_dataset)
    a = image_dataset.reshape((image_dataset.shape[0], 1, 512, 512))
    b = mask_dataset.reshape((mask_dataset.shape[0], 1, 512, 512))

    x_train, x_test, y_train, y_test = train_test_split(a, b, test_size=0.3, random_state=0)
    x_train =batch_numpy(x_train,batch)
    x_test =batch_numpy(x_test,batch)
    y_train = batch_numpy(y_train,batch)
    y_test = batch_numpy(y_test,batch)

    list_1 = zip(x_train, y_train)
    list_2 = zip(x_test, y_test)
    return list(list_1), list(list_2)

def batch_numpy(numpy, batchsize):

    list_new = []
    length = numpy.shape[0]
    if length % batchsize == 0:
        loop = int(length / batchsize)
    else:
        loop = int(length / batchsize) + 1

    for i in range(loop):
        if i == loop - 1:
            list_new.append(numpy[i * batchsize:,:,:,:])
        else:

            list_new.append(numpy[i * batchsize:(i+1) * batchsize, :, :, :])

    return list_new






#no use
def get_new_testdata():
    path1 = 'C:/PycharmProjects/Retinal-Vessel-Segmentation-using-variants-of-UNET/test/image'
    path2 = 'C:/PycharmProjects/Retinal-Vessel-Segmentation-using-variants-of-UNET/test/mask'


    image_dataset = []
    mask_dataset = []

    patch_size = 512

    images = sorted(os.listdir(path1))
    for i, image_name in enumerate(images):
        a = singel_image_preprocessing(image_name, path1, patch_size)
        a = np.array(a)
        a = a.reshape((a.shape[0],1,512, 512))
        image_dataset.append(a)

    masks = sorted(os.listdir(path2))
    for i, mask_name in enumerate(masks):
        a = singel_image_preprocessing(mask_name,path2,patch_size)
        a = np.array(a)
        a = a.reshape((a.shape[0], 1, 512, 512))
        mask_dataset.append(a)


    return image_dataset,mask_dataset


#no use
def convert_back(image):
    list_image = []
    for i in range(8):
        list_summe = []
        for j in range(8):
            list_summe.append(image[i*8+j,0,:,:])
        list_image.append(np.concatenate(list_summe, axis=1))

    return np.concatenate(list_image)

