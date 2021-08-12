import sys

import numpy as np
import cv2
from openvino.inference_engine import IENetwork, IECore

# C++ module for extracting pose from PAFs and heatmaps
from pose_extractor import extract_poses

limbIds = [
        [ 1,  2], [ 1,  5], [ 2,  3], [ 3,  4], [ 5,  6], [ 6,  7], [ 1,  8], [ 8,  9], [ 9, 10], [ 1, 11],
        [11, 12], [12, 13], [ 1,  0], [ 0, 14], [14, 16], [ 0, 15], [15, 17], [ 2, 16], [ 5, 17] ]

limbColors = [
    (255,  0,  0), (255, 85,  0), (255,170,  0),
    (255,255,  0), (170,255,  0), ( 85,255,  0),
    (  0,255,  0), (  0,255, 85), (  0,255,170),
    (  0,255,255), (  0,170,255), (  0, 85,255),
    (  0,  0,255), ( 85,  0,255), (170,  0,255),
    (255,  0,255), (255,  0,170), (255,  0, 85)
]

def renderPeople(img, people, scaleFactor=4, threshold=0.5):
    global limbIDs
    global limbColors
    # 57x32 = resolution of HM and PAF
    scalex = img.shape[1]/(57 * scaleFactor)
    scaley = img.shape[0]/(32 * scaleFactor)
    for person in people:
        for i, limbId in enumerate(limbIds[:-2]):
            x1, y1, conf1 = person[limbId[0]*3:limbId[0]*3+2 +1]
            x2, y2, conf2 = person[limbId[1]*3:limbId[1]*3+2 +1]
            if conf1>threshold and conf2>threshold:
                cv2.line(img, (int(x1*scalex),int(y1*scaley)), (int(x2*scalex),int(y2*scaley)), limbColors[i], 2)


def main():

    # Prep for OpenVINO Inference Engine for human pose estimation
    ie = IECore()
    model_hp = 'intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001'
    net_hp  = ie.read_network(model=model_hp+'.xml', weights=model_hp+'.bin')
    input_name_hp   = next(iter(net_hp.input_info))                     # Input blob name "data"
    input_shape_hp  = net_hp.input_info[input_name_hp].tensor_desc.dims # [1,3,256,456]
    PAF_blobName    = list(net_hp.outputs.keys())[0]                    # 'Mconv7_stage2_L1'
    HM_blobName     = list(net_hp.outputs.keys())[1]                    # 'Mconv7_stage2_L2'
    PAF_shape       = net_hp.outputs[PAF_blobName].shape                #  [1,38,32,57] 
    HM_shape        = net_hp.outputs[HM_blobName].shape                 #  [1,19,32,57]
    exec_net_hp     = ie.load_network(net_hp, 'CPU')

    # Open a USB webcam
    cam = cv2.VideoCapture(0)
    #cam = cv2.VideoCapture('people.264')
    if cam.isOpened()==False:
        print('Failed to open the input movie file (or a webCam)')
        sys.exit(-1)

    while cv2.waitKey(1) != 27:     # 27 == ESC

        ret, img = cam.read()
        if ret==False:
            return 0

        inblob = cv2.resize(img, (input_shape_hp[3], input_shape_hp[2]))    # 3=Width, 2=Height
        inblob = inblob.transpose((2, 0, 1))                                # Change data layout from HWC to CHW
        inblob = inblob.reshape(input_shape_hp)

        res_hp = exec_net_hp.infer(inputs={input_name_hp: inblob})          # Infer poses

        heatmaps = res_hp[HM_blobName ][0]
        PAFs     = res_hp[PAF_blobName][0]
        people = extract_poses(heatmaps[:-1], PAFs, 4)                      # Construct poses from HMs and PAFs

        renderPeople(img, people, 4, 0.2)
        cv2.imshow('Result', img)

    cv2.destroyAllWindows()
    return 0

if __name__ == '__main__':
    sys.exit(main())

