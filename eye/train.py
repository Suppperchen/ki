from torch import optim
import torch
import math
from torch.autograd import Variable
import diceloss
import unet2d
from loaddata import get_listData


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

    list_train,list_val  = get_listData('Data/train/image', 'Data/train/mask', 1)

    model = unet2d.unet_2d(1, 1)
    model.cuda()
    path = "eye.pth"

    train(800000, model, list_train, list_val, path)



start_training()