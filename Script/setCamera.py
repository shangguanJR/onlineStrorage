import numpy as np


def setupCamera():
    bounds = [0] * 6
    getNode("0001").GetBounds(bounds)
    bounds = np.array(bounds)
    center = (bounds[::2] + bounds[1::2]) / 2
    viewUp = [0, 0, 1]
    focalPoint = np.copy(center)
    position = np.copy(center)
    position[1] += 500 # 朝Anterior移动500mm
    print("CameraPosition:", position)
    cameraNode = getNode("Camera")
    cameraNode.SetViewUp(viewUp)
    cameraNode.SetFocalPoint(focalPoint)
    cameraNode.SetPosition(position)
    cameraNode.ResetClippingRange()
    threeDView = slicer.app.layoutManager().threeDWidget(0).threeDView()
    threeDView.setFixedSize(1024, 1024)



def setupBackground():
    threeDView = slicer.app.layoutManager().threeDWidget(0).threeDView()
    viewNode = threeDView.mrmlViewNode()
    viewNode.SetBoxVisible(0)
    viewNode.SetBackgroundColor(0,0,0)
    viewNode.SetBackgroundColor2(0, 0, 0)
    viewNode.SetAxisLabelsVisible(0)