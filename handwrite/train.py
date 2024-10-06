
from torch import optim
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import model
from loaddata import get_data_mnist


def train(num_epochs, cnn_model, list_train, list_val, path_save_model):

    cnn_model.train()
    #cnn_model.load_state_dict(torch.load('test.pth'))
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    #loss_func = nn.L1Loss()
    loss_func = nn.CrossEntropyLoss()
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
            #labels_train = Variable(torch.Tensor(labels_train).type(torch.int64))
            labels_train =labels_train.cuda()



            output = cnn_model(images_train)
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
             #labels_val = Variable(torch.Tensor(labels_val).type(torch.int64))
             labels_val = Variable(torch.Tensor(labels_val).type(torch.int64))
             labels_val =labels_val.cuda()

             output = cnn_model(images_val)
             loss_val = loss_func(output, labels_val)



             loss_summe_val = loss_summe_val+ float(loss_val)

         loss_summe_val = loss_summe_val/(i+1)
         print('Epoch [{}/{}], Loss_val: {:.4f}'
              .format(epoch + 1, num_epochs, loss_summe_val))



         if loss_summe_val<loss_update:
              torch.save(cnn_model.state_dict(), path_save_model)
              print('model saving in Epoch [{}/{}], val_Loss:{:.4f}'.
                     format(epoch + 1, num_epochs, loss_summe_val))
              loss_update = loss_summe_val


         print('Loss_best_val: {:.4f}'.format(loss_update))

         pass

    pass

if __name__ == "__main__":
    list1,list2 = get_data_mnist()
    model = model.cnnSimple()
    model.cuda()



    train(2000,model,list1,list2,"test_without_transform.pth")








