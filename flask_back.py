import base64
import numpy as np
import torch
from torch.autograd import Variable
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from unet2d import unet_2d
from handwrite.model import cnnSimple
from flask_cors import CORS
from matplotlib import pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

app = Flask(__name__)
CORS(app)
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
@app.route("/test")
def hello():
    return "<p>H!</p>"

@app.route('/handwrite', methods=['POST'])
def handwrite():
    content = request.json
    imgdata = base64.b64decode(str(content['image']))
    image_stream = BytesIO(imgdata)
    image = Image.open(image_stream)
    # the image in ui need be resize to 28
    image = image.resize((28, 28))
    np_img = np.array(image)

    img_numpy = np_img.astype(float)
    img_numpy = img_numpy / 255
    img_numpy = img_numpy[:,:,1]
    img_numpy = np.reshape(img_numpy, (1, 1, 28, 28))

    #

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.MNIST(root='handwrite/data', train=False, transform=transforms.ToTensor())
    train_dataloader = DataLoader(trainset, batch_size=64)

    for batch_idx, samples in enumerate(train_dataloader):
        if batch_idx>0:
            break
        a,_ = samples

    #

    with torch.no_grad():
        data = Variable(torch.Tensor(img_numpy))
        data = data.cuda()

        a = Variable(torch.Tensor(a))
        a = a.cuda()

        cnnmini = cnnSimple()
        cnnmini.eval()
        cnnmini.cuda()
        cnnmini.load_state_dict(torch.load('handwrite/test.pth'))
        out = cnnmini(data)

        out = torch.max(out, 1)
    print(out)

    return jsonify({"KIImage": "sucesse"})

@app.route('/post-retinal-vessel', methods=['POST'])

def post_retinal_vessel():
    content = request.json
    imgdata = base64.b64decode(str(content['image']))
    image_stream = BytesIO(imgdata)
    image = Image.open(image_stream)
    np_img = np.array(image)

    img_numpy = np_img.astype(float)
    img_numpy = img_numpy / 255
    img_numpy = img_numpy[:,:,1]
    img_numpy = np.reshape(img_numpy, (1, 1, 512, 512))
    print(img_numpy.shape)

    with torch.no_grad():
        data = Variable(torch.Tensor(img_numpy))
        data = data.cuda()

        unet = unet_2d(1, 1)
        unet.cuda()
        unet.load_state_dict(torch.load('eye/test13.pth'))
        out = unet(data)

    np_data = out.cpu().data.numpy()
    np_data = np.reshape(np_data, (512, 512))


    np_data = np_data*255
    np_data = np_data.astype(np.uint8)

    buffer = BytesIO()
    sendback = Image.fromarray(np_data)
    sendback.save(buffer, format="PNG")

    return jsonify({"KIImage": str(base64.b64encode(buffer.getvalue()).decode("utf-8")) })


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(host="0.0.0.0", port=5000)