from torch import optim
import torch
import math
from torch.autograd import Variable
import diceloss
import torch.nn as nn
import unet2d
from unetGitHub import  U_Net
from loaddata import get_listData,get_new_data


def train(num_epochs, unet, list_train,list_val,path_save_model):

    unet.train()

    #unet.load_state_dict(torch.load('eye.pth'))

    optimizer = optim.Adam(unet.parameters(), lr=0.001)
    #loss_func = nn.L1Loss()
    #loss_func = nn.MSELoss()
    loss_func = diceloss.DiceLoss()
    loss_update = math.inf
    for epoch in range(num_epochs):
        loss_summe = 0
        loss_summe_val =0

        for i, data_label in enumerate(list_train,1):
            images_train,labels_train= data_label
            optimizer.zero_grad()
            images_train = Variable(torch.Tensor(images_train))
            images_train =images_train.cuda()
            labels_train = Variable(torch.Tensor(labels_train))
            labels_train =labels_train.cuda()



            output = unet(images_train)
            loss = loss_func(output, labels_train)

            loss.backward()
            optimizer.step()
            loss_summe = loss_summe+ float(loss)
            if i % 10 == 0:
                print(float(loss))

        loss_summe = loss_summe/(i+1)
        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, loss_summe))

        with torch.no_grad():

         for i, data_label in enumerate(list_val,1):

             images_val,labels_val= data_label
             images_val = Variable(torch.Tensor(images_val))
             images_val =images_val.cuda()
             labels_val = Variable(torch.Tensor(labels_val))
             labels_val =labels_val.cuda()

             output = unet(images_val)
             loss_val = loss_func(output, labels_val)



             loss_summe_val = loss_summe_val+ float(loss_val)

         loss_summe_val = loss_summe_val/(i+1)
         print('Epoch [{}/{}], Loss_val: {:.4f}'
              .format(epoch + 1, num_epochs, loss_summe_val))



         if loss_summe_val<loss_update:
              torch.save(unet.state_dict(), path_save_model)
              print('model saving in Epoch [{}/{}], val_Loss:{:.4f}'.
                     format(epoch + 1, num_epochs, loss_summe_val))
              loss_update = loss_summe_val


         print('Loss_best_val: {:.4f}'.format(loss_update))

         pass

    pass


def start_training():
    #list_train = get_listData('Data/train/image', 'Data/train/mask', 8)
    #list_label = get_listData('Data/test/image', 'Data/test/mask', 8)
    list_train,list_val = get_new_data(16)
    model = unet2d.unet_2d(1, 1)
    model.cuda()
    path = "test9.pth"

    train(800000, model, list_train, list_val, path)


#train with only one image
def testonlyOne():
    list_val = get_listData('Data/train/image', 'Data/train/mask', 4)
    a,b = list_val[0]
    a = a[0,:,:,:]
    b = b[0,:,:,:]
    a = a.reshape(1,3,128,128)
    b = b.reshape(1,1,128,128)
    model = unet2d.unet_2d(3, 1)
    model.cuda()
    path = "test.pth"
    train(4000, model, [(a,b)], [(a,b)], path)


start_training()