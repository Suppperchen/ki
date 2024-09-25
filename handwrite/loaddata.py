import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


#60000,10000
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,transform=transforms.ToTensor())
#testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
#trainset = torchvision.datasets.EMNIST(root='./data',split='letters', train=True, download=False)
train_dataloader = DataLoader(trainset, batch_size=64, shuffle=True)
#for batch_idx, samples in enumerate(train_dataloader):
#loss = nn.CrossEntropyLoss()















