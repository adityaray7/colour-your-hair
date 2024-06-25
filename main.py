import time
import cv2
import keras
import numpy as np
import os


def predict(image, model, height=256, width=256):
    im = image / 255
    im = cv2.resize(im, (height, width))
    im = im.reshape((1,) + im.shape)
    # print(im.shape)
    model.setInput(im)
    pred = model.forward()

    return pred[0]

def predict_border(frame, mask, model_border, height=256, width=256):
    im = frame / 255
    im = cv2.resize(im, (height, width))
    st = np.dstack((im, mask))
    # print(st.shape)
    st = st.reshape((1,) + st.shape)
    # print(st.shape)
    model_border.setInput(st)
    pred = model_border.forward()

    return pred[0]

def transfer(image, mask):
    # theresholding the the mask image
    mask[mask > 0.40] = 255
    mask[mask <= 0.40] = 0

    # resizing the mask image to size of the input image
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # creating the numpy array of size of the input image
    mask_n = np.zeros_like(image)

    # saving the mask image in the all three channels
    mask_n[:, :, 0] = mask
    mask_n[:, :, 1] = mask
    mask_n[:, :, 2] = mask
    # spliting the mask_n image image in three channels
    (B, G, R) = cv2.split(mask_n)

    # assigning the B,G R value in the mask_n image to get colored mask image
    B[(mask == 255)] = 35  # blue channel value
    G[(mask == 255)] = 2  # green channel value
    R[(mask == 255)] = 250  # red channel value

    # merging the channels in single masked image
    masked = cv2.merge([B, G, R])

    # cv2.imwrite(os.path.join("Segmentation/chowis_data/", "1_"+path_image.split('/')[-1]), masked)

    # blending the masked image with input image
    alpha = 0.8
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(image, alpha, masked, beta, 0.0)
    return dst


if __name__ == '__main__':

    camera = cv2.VideoCapture(0)  # 0 means read from local camera
    model = cv2.dnn.readNetFromONNX('samp-dist-v5.onnx')
    model_border = cv2.dnn.readNetFromONNX('samp-bor-v5.onnx')

    while True:
        start = time.time()
        ret, frame = camera.read()

        if ret:
            h = time.time()
            mask = predict(frame, model)
            pred = predict_border(frame, mask, model_border)
            pred = transfer(frame, pred)
            cv2.imshow('frame', pred)
            

            if cv2.waitKey(1) & 0xFF == ord(chr(27)):
                break
    camera.release()