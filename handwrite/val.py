from torch.autograd import Variable
import torch
import model
from matplotlib import pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
def show_out_Image(data_np,lable_np,path):

    with torch.no_grad():
        data = Variable(torch.Tensor(data_np))
        data = data.cuda()
        cnnMini = model.cnnSimple()
        cnnMini.cuda()
        cnnMini.load_state_dict(torch.load(path))
        out = cnnMini(data)
        print(out.shape)
        out = torch.max(out,1)


    #np_data = out.cpu().data.numpy()

    print(lable_np)
    print(out)
    plt.imshow(data_np[15,:,:,:].reshape(28,28), cmap='gray')
    plt.show()

    return

if __name__ == "__main__":
    batch_size = 64
    # 128, do later

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    train_dataloader = DataLoader(trainset, batch_size=batch_size)

    for batch_idx, samples in enumerate(train_dataloader):

        if batch_idx>0:
            break
        a,b = samples
        show_out_Image(a,b,"test.pth")
