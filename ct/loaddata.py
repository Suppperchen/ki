import SimpleITK as sitk
import numpy as np

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