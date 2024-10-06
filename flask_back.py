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
import cv2
from torchvision.transforms import v2
from skimage.transform import rescale,resize,rotate

app = Flask(__name__)
CORS(app)
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
@app.route("/test")
def hello():
    return "<p>H!</p>"



def rotate_image(image):
    range = np.arange(-7.5,7.5,0.1)
    np.random.seed(10)
    rotate_angel =np.random.choice(range, 1)
    image = rotate(image,float(rotate_angel))
    return image
def transform(img):
    transforms = v2.Compose([
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    out = transforms(img)
    plt.figure()
    f, axarr = plt.subplots(2, 1)
    axarr[0].imshow(img)
    axarr[1].imshow(out)
    plt.show()

    return out
def resize_mirror_rotate(img):

    img_resize = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR)[:,:,0]

    img_mirror = np.flip(img_resize, 1)

    img_rotate = rotate_image(img_resize)

    img_resize = np.reshape(img_resize, (1, 1, 28, 28))

    img_mirror = np.reshape(img_mirror, (1, 1, 28, 28))

    img_rotate = np.reshape(img_rotate, (1, 1, 28, 28))


    return np.concatenate((img_resize, img_mirror, img_rotate), axis=0)


@app.route('/handwrite', methods=['POST'])
def handwrite():
    content = request.json
    imgdata = base64.b64decode(str(content['image']))
    image_stream = BytesIO(imgdata)
    image_original = Image.open(image_stream)

    # the image in ui need be resize to 28
    image = image_original.resize((28, 28))

    #image_rescaled = rescale(np.array(image_original), 28)
    np_img = np.array(image)


    img_numpy = np_img.astype(float)
    img_numpy = img_numpy / 255
    img_numpy = img_numpy[:,:,0]


    img_numpy = np.reshape(img_numpy, (1, 1, 28, 28))

    with torch.no_grad():
        data = Variable(torch.Tensor(img_numpy))
        data = data.cuda()

        cnnmini = cnnSimple()
        cnnmini.eval()
        cnnmini.cuda()
        cnnmini.load_state_dict(torch.load('handwrite/test_without_transform.pth'))
        out = cnnmini(data)


    print(torch.max(out, 1).indices.cpu().data.numpy())

    return jsonify({"KIhandwrite": str(torch.max(out, 1).indices.cpu().data.numpy())})
    #return jsonify({"KIhandwrite": "sure"})

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
        unet.load_state_dict(torch.load('eye/eye.pth'))
        out = unet(data)

    np_data = out.cpu().data.numpy()
    np_data = np.reshape(np_data, (512, 512))


    np_data = np_data*255
    np_data = np_data.astype(np.uint8)

    buffer = BytesIO()
    sendback = Image.fromarray(np_data)
    sendback.save(buffer, format="PNG")

    return jsonify({"KIImage": str(base64.b64encode(buffer.getvalue()).decode("utf-8")) })


@app.route('/post-ct', methods=['POST'])

def post_ct():
    content = request.json
    imgdata = base64.b64decode(str(content['image']))
    image_stream = BytesIO(imgdata)
    image = Image.open(image_stream)
    np_img = np.array(image)

    img_numpy = np_img.astype(float)
    img_numpy = img_numpy / 255
    img_numpy = img_numpy[:,:,1]
    img_numpy = np.reshape(img_numpy, (1, 1, 256, 256))
    print(img_numpy.shape)

    with torch.no_grad():
        data = Variable(torch.Tensor(img_numpy))
        data = data.cuda()

        unet = unet_2d(1, 1)
        unet.cuda()
        unet.load_state_dict(torch.load('ct/ct.pth'))
        out = unet(data)

    np_data = out.cpu().data.numpy()
    np_data = np.reshape(np_data, (256, 256))


    np_data = np_data*255
    np_data = np_data.astype(np.uint8)

    buffer = BytesIO()
    sendback = Image.fromarray(np_data)
    sendback.save(buffer, format="PNG")

    return jsonify({"KIImage": str(base64.b64encode(buffer.getvalue()).decode("utf-8")) })

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(host="0.0.0.0", port=5000)