import SimpleITK as sitk
import numpy as np
from PIL import Image
def load_data(path_mr,path_ct,path_mask):

    inputImage_mr = sitk.ReadImage(path_mr)
    inputImage_ct = sitk.ReadImage(path_ct)
    inputImage_mask = sitk.ReadImage(path_mask)

    np_mr = sitk.GetArrayFromImage(inputImage_mr)
    np_mr = np_mr.astype(dtype=np.float32)


    np_ct = sitk.GetArrayFromImage(inputImage_ct)
    np_ct = np_ct.astype(dtype=np.float32)


    np_mask = sitk.GetArrayFromImage(inputImage_mask)
    np_mask = np_mask.astype(dtype=np.float32)


    np_mr = np_mr-np.amin(np_mr)
    np_mr = np_mr * np_mask
    np_mr = np_mr / 4095


    np_ct = np_ct - np.amin(np_ct)
    np_ct = np_ct*np_mask
    np_ct = np_ct / 4095


    new_shape_np_mr = np.reshape(np_mr, ( np_mr.shape[0],1, 256, 256))
    new_shape_np_ct = np.reshape(np_ct, ( np_mr.shape[0],1, 256, 256))



    return new_shape_np_mr, new_shape_np_ct




def get_alldata(list,random = True):
    list_data = []
    list_label = []

    for i in list:

        path_ct = 'data_ct/new_ct/' + str(i) + '.gipl'
        path_mr = 'data_ct/new_mr/' + str(i) + '.gipl'
        path_mask = 'data_ct/mask_drop/' + str(i) + '.gipl'

        a, b = load_data(path_mr, path_ct, path_mask)

        list_data.append(a)
        list_label.append(b)

    data = np.concatenate(list_data, axis=0)
    label = np.concatenate(list_label, axis=0)

    if random :
        seed = np.random.randint(0, 10000)
        np.random.seed(seed)
        np.random.shuffle(data)
        np.random.seed(seed)
        np.random.shuffle(label)

    return data, label
def change_data_asList(data,label,batchsize):
    if data.shape[0]%batchsize== 0:
        nur = data.shape[0]//batchsize
    else:
        nur = data.shape[0]//batchsize +1

    list = []
    for i in range(nur):
        index_first = i*batchsize
        index_last = (i+1)*batchsize
        list.append((data[index_first:index_last,:,:,:],label[index_first:index_last,:,:,:]))
    return list

def createpngData(image,image_mask,image_label,indexNur):

    image = sitk.ReadImage(image)
    image = sitk.GetArrayFromImage(image)
    image = image.astype(dtype=np.float32)


    image_mask = sitk.ReadImage(image_mask)
    image_mask = sitk.GetArrayFromImage(image_mask)
    image_mask = image_mask.astype(dtype=np.float32)

    image_label = sitk.ReadImage(image_label)
    image_label = sitk.GetArrayFromImage(image_label)
    image_label = image_label.astype(dtype=np.float32)

    image = image - np.amin(image)
    image = image * image_mask
    image = image / 4095

    image_label = image_label - np.amin(image_label)
    image_label = image_label * image_mask
    image_label = image_label / 4095



    image = image*255
    image = image.astype(np.uint8)

    image_label = image_label * 255
    image_label = image_label.astype(np.uint8)

    for i in range(image.shape[0]):
        save = Image.fromarray(image[i,:,:])
        save.save('creatPNG/val/data/MRT'+indexNur+'_' + str(i) + '.png')

        save = Image.fromarray(image_label[i, :, :])
        save.save('creatPNG/val/label/MRT'+indexNur+'_' + str(i) + '.png')






    return


if __name__ == "__main__":

    # list = [0, 1, 2, 4, 7, 8, 9, 10, 12, 13]
    # list = [5,6,14,15]
    #list = [3, 11]

    # for i in list:
    #     path_ct = 'data_ct/new_ct/' + str(i) + '.gipl'
    #     path_mr = 'data_ct/new_mr/' + str(i) + '.gipl'
    #     path_mask = 'data_ct/mask_drop/' + str(i) + '.gipl'
    #     createpngData(path_mr, path_mask, path_ct, str(i))

    for i in range(10):
        print(i)
        i = i+3
        print(i)

