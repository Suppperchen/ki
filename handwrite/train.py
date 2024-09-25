import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import model


def train(num_epochs, cnn_model, list_train, list_val, path_save_model):

    cnn_model.train()
    #unet.load_state_dict(torch.load('eye.pth'))
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
             labels_val = Variable(torch.Tensor(labels_val))
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
    batch_size = 1
    #128, do later

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor())
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    model = model.cnnSimple()
    model.cuda()


    list1 = []
    list2 = []
    for batch_idx, samples in enumerate(train_dataloader):
        nurForVal = (60000 / batch_size)*5/6
        if batch_idx>nurForVal:
            list2.append(samples)
        else:
            list1.append(samples)


    train(2020,model,list1,list2,"test1.pth")








