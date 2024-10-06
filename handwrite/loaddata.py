import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



def load_minidata():
    list_data = []
    list_label = []

    list_label_val = []
    list_data_val = []

    for i in range(10):
        filename = 'datamini/handwrite'+str(i)+'.npy'
        with open(filename, 'rb') as file:
            handwrite_numpy_data = np.load(file)


            handwrite_numpy_data = handwrite_numpy_data.astype(np.float32)
            handwrite_numpy_data = handwrite_numpy_data / 255

            list_original = []


            for j in range(10):
                list_original.append(np.reshape(handwrite_numpy_data[3*i,:,:,:], (1, 1, 28,28)))
            handwrite_numpy_data = np.concatenate(list_original)




            list_data.append(handwrite_numpy_data[:8,:,:,:])
            list_label.append(np.full((8 ), i))

            list_data_val.append(handwrite_numpy_data[8:, :, :, :])
            list_label_val.append(np.full((2), i))



    return np.concatenate(list_data), np.concatenate(list_label),np.concatenate(list_data_val),np.concatenate(list_label_val)


def randomdata(data,label):
    seed = np.random.randint(0, 10000)
    np.random.seed(seed)
    np.random.shuffle(data)
    np.random.seed(seed)
    np.random.shuffle(label)

    return data, label

def get_data():
    a, b, c, d = load_minidata()
    c, d = randomdata(c, d)
    a, b = randomdata(a, b)

    list_summe = []
    list_summe_val = []

    for i in range(4):
        list_summe.append((a[20 * i:20 * (i + 1), :, :, :], b[20 * i:20 * (i + 1)]))

    list_summe_val.append((c, d))
    return list_summe, list_summe_val

def get_data_mnist():
    batch_size = 64
    # 128, do later

    #transform = transforms.Compose(
    #    [transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor())
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)


    list1 = []
    list2 = []
    for batch_idx, samples in enumerate(train_dataloader):
        nurForVal = (60000 / batch_size) * 5 / 6
        if batch_idx > nurForVal:
            list2.append(samples)
        else:
            list1.append(samples)
    return list1, list2
























