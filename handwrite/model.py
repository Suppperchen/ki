import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
class cnnSimple(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.b1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, padding=1,stride=2)
        self.b3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1,stride=2)
        self.b4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2)
        self.b5 = nn.BatchNorm2d(64)

        self.l1 = nn.Linear(64, 128)
        self.l2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.b1(x)
        x = F.relu(x,inplace=True)

        x = self.conv2(x)
        x = self.b2(x)
        x = F.relu(x,inplace=True)

        x = self.conv3(x)
        x = self.b3(x)
        x = F.relu(x, inplace=True)

        x = nn.MaxPool2d(2)(x)
        x = nn.Dropout(p=0.4)(x)

        x = self.conv4(x)
        x = self.b4(x)
        x = F.relu(x, inplace=True)

        x = nn.MaxPool2d(2)(x)

        x = self.conv5(x)
        x = self.b5(x)
        x = F.relu(x, inplace=True)

        x = nn.Dropout(p=0.4)(x)

        x = nn.Flatten()(x)

        x = self.l1(x)
        x = F.relu(x, inplace=True)
        x = nn.Dropout(p=0.4)(x)

        x = self.l2(x)
        x = F.log_softmax(x,dim=1)
        return x
if __name__ == "__main__":
    model = cnnSimple()
    model.cuda()
    summary(model, (1, 28, 28))

# model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', padding="same",
#         kernel_initializer='he_normal',input_shape=(IMG_ROWS, IMG_COLS, 1)))
#
# model.add(BatchNormalization())
#
# model.add(Conv2D(32,kernel_size=(3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(BatchNormalization())
# # Add dropouts to the model
# model.add(Dropout(0.4))
# model.add(Conv2D(64, kernel_size=(3, 3), strides=2,padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())
# model.add(Conv2D(64, kernel_size=(3, 3), strides=2,padding='same', activation='relu'))
# # Add dropouts to the model
# model.add(Dropout(0.4))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# # Add dropouts to the model
# model.add(Dropout(0.4))
# model.add(Dense(NUM_CLASSES, activation='softmax'))