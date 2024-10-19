from torch.autograd import Variable
import torch
import unet2d
from matplotlib import pyplot as plt
import loaddata
def show_out_Image(data_np,lable_np,path,savepath1,savepath2):

    with torch.no_grad():
        data = Variable(torch.Tensor(data_np))
        data = data.cuda()

        #mask = Variable(torch.Tensor(lable_np))
        #mask = mask.cuda()

        unet = unet2d.unet_2d(1, 1)
        unet.cuda()
        unet.load_state_dict(torch.load(path))
        out = unet(data)


    np_data = out.cpu().data.numpy()

    plt.figure()
    f, axarr = plt.subplots(2, 1)
    axarr[0].imshow(np_data.reshape(256,256), cmap='gray')
    axarr[1].imshow(lable_np.reshape(256,256), cmap='gray')
    plt.savefig('saveResultPNG/val/'+savepath1 + '_' + savepath2 + '.png')



    return


if __name__ == "__main__":

    #list = [5,6,14,15]
    #list = [0, 1, 2, 4, 7, 8, 9, 10, 12, 13]
    list = [3,11]

    for ele in list:
        data, label = loaddata.get_alldata([ele], False)
        nur = data.shape[0]
        i = 0
        while i <nur:
            show_out_Image(data[i, :, :, :].reshape((1, 1, 256, 256)), label[i, :, :, :].reshape((1, 1, 256, 256)),
                           "ct.pth",str(ele),str(i))
            i = i + 4














