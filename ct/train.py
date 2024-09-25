import unet2d
from torch import optim
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import loaddata

def train(num_epochs, unet, list_train,list_val,path_save_model):

    unet.train()
    #unet.load_state_dict(torch.load('eye.pth'))
    optimizer = optim.Adam(unet.parameters(), lr=0.001)
    #loss_func = nn.L1Loss()
    loss_func = nn.MSELoss()
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

if __name__ == "__main__":
    list = [0, 1, 2, 4, 7, 8, 9, 10, 12, 13]
    list_val = [3, 11]
    batchsize = 5
    # list_test = [5,6,14,15]

    data_train, label_train = loaddata.get_alldata(list)
    data_val, label_val = loaddata.get_alldata(list_val)
    train_list = loaddata.change_data_asList(data_train, label_train, batchsize)
    val_list = loaddata.change_data_asList(data_val, label_val, batchsize)

    model = unet2d.unet_2d(1, 1)
    model.cuda()
    path = "test1.pth"

    a,b = train_list[0]
    print(a.shape)

    #train(800, model, train_list, val_list, path)





