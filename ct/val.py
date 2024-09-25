from torch.autograd import Variable
import torch
import unet2d
from matplotlib import pyplot as plt
import loaddata
def show_out_Image(data_np,lable_np,path):

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
    f, axarr = plt.subplots(3, 1)
    axarr[0].imshow(np_data.reshape(256,256), cmap='gray')
    axarr[1].imshow(data_np.reshape(256,256), cmap='gray')
    axarr[2].imshow(lable_np.reshape(256,256), cmap='gray')
    plt.show()

    return


if __name__ == "__main__":

    list = [5,6,14,15]

    data, label = loaddata.get_alldata(list,False)


    show_out_Image(data[12,:,:,:].reshape((1,1,256,256)),label[12,:,:,:].reshape((1,1,256,256)),"test1.pth")

