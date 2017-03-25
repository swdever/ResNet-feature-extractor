# Make sure YOU_PATH_TO_cuda-7.5/lib64 is added to LD_LIBRARY_PATH,
# because it seems that this specific version of caffe need cuda 7.5
import numpy as np
import sys
caffe_root = "caffe/" # Compile the specific version of caffe using by ResNet, replace your path here
sys.path.insert(0, caffe_root+ 'python')
import caffe
import cv2

if __name__ == "__main__":
    mean_file = "ResNet-50/ResNet_mean.binaryproto" # Replace your path here. Some code below need same process

    mean_blob = caffe.proto.caffe_pb2.BlobProto()
    mean_blob.ParseFromString(open(mean_file, "rb").read())
    # mean_npy has shape 1x3x224x224, more detail see https://github.com/KaimingHe/deep-residual-networks/issues/5
    mean_npy = caffe.io.blobproto_to_array(mean_blob)

    caffe.set_mode_gpu()
    model_def = "ResNet-50/ResNet-50-deploy.prototxt"
    model_weight = "ResNet-50/ResNet-50-model.caffemodel"

    net = caffe.Net(model_def, model_weight, caffe.TEST)

    batch_size = 1
    net.blobs['data'].reshape(batch_size, 3, 224, 224)
    img = cv2.imread("cat.jpg")
    img = cv2.resize(img, (224, 224))
    img = np.transpose(img, [2, 0, 1])

    net.blobs['data'].data[...] = img - mean_npy

    output = net.forward()

    # print all blob and its data shape
    for layer_name, blob in net.blobs.iteritems():
        print layer_name + "\t" + str(blob.data.shape)

    feat = net.blobs['pool5'].data # we will get feature after at least one forward
    print np.shape(feat)


