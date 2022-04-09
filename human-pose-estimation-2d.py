import sys

import numpy as np
import cv2

from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
from openvino.runtime import Core, Layout, Type

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
    core = Core()
    model_hp = 'intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml'
    net_hp  = core.read_model(model=model_hp)
    input_name_hp   = net_hp.input().get_any_name()        # Input blob name "data"
    input_shape_hp  = net_hp.input().get_shape()           # [1,3,256,456]
    PAF_blobName    = net_hp.output(0).get_any_name()      # 'Mconv7_stage2_L1'
    HM_blobName     = net_hp.output(1).get_any_name()      # 'Mconv7_stage2_L2'
    PAF_shape       = net_hp.output(0).get_shape()         #  [1,38,32,57] 
    HM_shape        = net_hp.output(1).get_shape()         #  [1,19,32,57]

    # Setup pre/postprocessor
    ppp = PrePostProcessor(net_hp)
    ppp.input().tensor().set_element_type(Type.u8).set_layout(Layout('NHWC')).set_spatial_dynamic_shape()
    ppp.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)
    ppp.input().model().set_layout(Layout('NCHW'))
    ppp.output(0).tensor().set_element_type(Type.f32)
    ppp.output(1).tensor().set_element_type(Type.f32)
    net_hp = ppp.build()

    exec_hp = core.compile_model(net_hp, 'CPU')
    ireq_hp = exec_hp.create_infer_request()

    #print(input_name_hp, input_shape_hp)
    #print(PAF_blobName, PAF_shape)
    #print(HM_blobName, HM_shape)

    # Open a USB webcam
    #cam = cv2.VideoCapture(0)
    cam = cv2.VideoCapture('people.264')
    if cam.isOpened()==False:
        print('Failed to open the input movie file (or a webCam)')
        sys.exit(-1)

    while cv2.waitKey(1) != 27:     # 27 == ESC

        ret, img = cam.read()
        if ret==False:
            return 0

        tensor = np.expand_dims(img, 0)
        res_hp = ireq_hp.infer({0: tensor})          # Infer poses

        heatmaps = ireq_hp.get_tensor(HM_blobName).data[0]
        PAFs     = ireq_hp.get_tensor(PAF_blobName).data[0]
        people = extract_poses(heatmaps[:-1], PAFs, 4)                      # Construct poses from HMs and PAFs

        renderPeople(img, people, 4, 0.2)
        cv2.imshow('Result', img)

    cv2.destroyAllWindows()
    return 0

if __name__ == '__main__':
    sys.exit(main())

