import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
class DoubleConv(nn.Module):#test for res net
    def __init__(self,in_channels,out_channels,mid_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(mid_channels,out_channels, kernel_size=3, padding=1)
        self.b1 = nn.BatchNorm2d(mid_channels)
        self.b2 = nn.BatchNorm2d(out_channels)

    def forward(self,x):

        x_conv1 = self.conv1(x)
        x_conv1 = self.b1(x_conv1)
        x_conv1 = F.relu(x_conv1,inplace=True)
        x_conv2 = self.conv2(x_conv1)
        x_conv2 = self.b2(x_conv2)
        x_conv2 = F.relu(x_conv2,inplace=True)
        x_conv2 = x_conv2+x_conv1

        return x_conv2

class DoubleConv1(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(mid_channels,out_channels, kernel_size=3, padding=1)
        self.b1 = nn.BatchNorm2d(mid_channels)
        self.b2 = nn.BatchNorm2d(out_channels)


    def forward(self,x):

        x = self.conv1(x)
        x = self.b1(x)
        x = F.relu(x,inplace=True)
        x = self.conv2(x)
        x = self.b2(x)
        x = F.relu(x,inplace=True)
        return x

class Up_Conv_copy_right(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.Tranconv = nn.ConvTranspose2d(in_channels,in_channels,kernel_size=3,stride=2,output_padding=(1,1))
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3)

    def forward(self,x,y):
        x = self.Tranconv(x)
        x = F.relu(x,inplace = True)
        x = self.conv(x)
        x = F.relu(x,inplace = True)
        x = torch.cat([F.pad(x, [1, 0, 0, 0]), y], dim=1)
        return  x

class Up_Conv_copy(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.Tranconv = nn.ConvTranspose2d(in_channels,in_channels,kernel_size=3,stride=2,output_padding=(1,1))
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3)
        self.b1 = nn.BatchNorm2d(out_channels)

    def forward(self,x,y):
        x = self.Tranconv(x)
        #x = F.relu(x,inplace = True)
        x = self.conv(x)
        x = self.b1(x)
        x = F.relu(x,inplace = True)
        x = torch.cat([x, y], dim=1)
        return  x

class Up_Conv_copy_left(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.Tranconv = nn.ConvTranspose2d(in_channels,in_channels,kernel_size=3,stride=2,output_padding=(1,1))
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3)

    def forward(self,x,y):
        x = self.Tranconv(x)
        x = F.relu(x,inplace = True)
        x = self.conv(x)
        x = F.relu(x,inplace = True)
        x = torch.cat([F.pad(x, [0, 0, 1, 0]), y], dim=1)
        return  x

class Output(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.output = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.output(x)



class unet_2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(unet_2d, self).__init__()
        self.down1 = DoubleConv1(in_channels, 16, 16)

        self.down2 = DoubleConv1(16, 32, 32)

        self.down3 = DoubleConv1(32, 64, 64)

        self.down4 = DoubleConv1(64, 128, 128)

        self.down5 = DoubleConv1(128, 256, 256)

        #self.latent_space = DoubleConv1(1024, 1024, 1024)

        #self.up5 = Up_Conv_copy(1024, 1024)
        #self.up_conv5 = DoubleConv1(2048, 1024, 1024)

        self.up4 = Up_Conv_copy(256, 128)
        self.up_conv4 = DoubleConv1(256, 128, 128)

        self.up3 = Up_Conv_copy(128, 64)
        self.up_conv3 = DoubleConv1(128, 64, 64)

        self.up2 = Up_Conv_copy(64, 32)
        self.up_conv2 = DoubleConv1(64, 32, 32)

        self.up1 = Up_Conv_copy(32, 16)
        self.up_conv1 = DoubleConv1(32, 16, 16)

        self.out = Output(16, out_channels)



    def forward(self, x):
        output_down1 = self.down1(x)
        output_down1_maxpooling = nn.MaxPool2d(2)(output_down1)

        output_down2 = self.down2(output_down1_maxpooling)
        output_down2_maxpooling = nn.MaxPool2d(2)(output_down2)

        output_down3 = self.down3(output_down2_maxpooling)
        output_down3_maxpooling = nn.MaxPool2d(2)(output_down3)

        output_down4 = self.down4(output_down3_maxpooling)
        output_down4_maxpooling = nn.MaxPool2d(2)(output_down4)

        output_down5 = self.down5(output_down4_maxpooling)
        #output_down5_maxpooling = nn.MaxPool2d(2)(output_down5)

        #output_latent_space = self.latent_space(output_down5_maxpooling)

        #output_up5 = self.up5(output_latent_space, output_down5)
        #output_up5_conv5 = self.up_conv5(output_up5)

        output_up4 = self.up4(output_down5, output_down4)
        output_up4_conv4 = self.up_conv4(output_up4)

        output_up3 = self.up3(output_up4_conv4, output_down3)
        output_up3_conv3 = self.up_conv3(output_up3)

        output_up2 = self.up2(output_up3_conv3, output_down2)
        output_up2_conv2 = self.up_conv2(output_up2)

        output_up1 = self.up1(output_up2_conv2, output_down1)
        output_up1_conv1 = self.up_conv1(output_up1)

        out = self.out(output_up1_conv1)

        return out

if __name__ == "__main__":
    model = unet_2d(3, 1)
    model.cuda()
    summary(model, (3, 512, 512))


