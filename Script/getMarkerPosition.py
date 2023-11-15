import h5py
import argparse
import numpy as np

points = np.array([
    [84.88449, 77.48327, -45.37695],
    [54.9133, 77.47672, -45.42195],
    [24.95245, 77.47912, -45.41777],
    [-5.01327, 77.49688, -45.15917],
    [-34.97945, 77.49976, -45.4615],
    [-64.94115, 77.50354, -45.53236],
    [84.90445, 107.38661, -35.26414],
    [54.91996, 107.38695, -35.28282],
    [24.93942, 107.39508, -35.24637],
    [-5.02827, 107.4052, -35.24358],
    [-35.0082, 107.39659, -35.30799],
    [-64.98358, 107.41427, -35.39882],
    [84.93766, 137.31649, -25.06133],
    [54.94122, 137.31768, -25.10021],
    [24.95253, 137.32427, -25.09061],
    [-5.03708, 137.3306, -25.14346],
    [-35.02202, 137.33263, -25.15919],
    [-65.02061, 137.33134, -25.22316],
    [84.94383, 167.28667, -24.910654],
    [54.94471, 167.30265, -24.89795],
    [24.9506, 167.30234, -24.98471],
    [-5.03476, 167.31112, -24.95841],
    [-35.03189, 167.31892, -25.00525],
    [-65.00871, 167.32395, -25.09241],
    [84.92648, 197.315, -34.78574],
    [54.94638, 197.32074, -34.75196],
    [24.96715, 197.32843, -34.6725],
    [-5.01695, 197.32937, -34.81415],
    [-34.99109, 197.33572, -34.83209],
    [-64.97059, 197.33803, -34.90707],
    [84.92983, 227.31644, -44.66769],
    [54.94406, 227.33247, -44.67105],
    [24.97355, 227.33178, -44.68214],
    [-4.99832, 227.33934, -44.68891],
    [-34.97363, 227.33686, -44.69023],
    [-64.9388, 227.34502, -44.80812],
])


def read_h5(fileName: str):
    with h5py.File(fileName, "r") as file:
        transform_matrix = file['TransformGroup/0/TransformParameters'][()]
        R = np.array(transform_matrix[:9]).reshape(3, 3)
        t = np.array(transform_matrix[9:]).reshape(3, 1)
        t = np.array((t[0], t[2], t[1]))
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3:] = t
        transform_matrix[np.abs(transform_matrix) < 1e-9] = 0
    return transform_matrix


def getWorldPos(transformFile):
    transform_matrix = read_h5(transformFile)
    print(transform_matrix)
    points_homo = np.ones((36, 4))
    points_homo[:, :3] = np.copy(points)
    worldPos = np.zeros((36, 3))
    for i in range(36):
        point = points_homo[i].reshape(4, 1)
        point = transform_matrix@point
        point /= point[3, 0]
        worldPos[i] = point.reshape(-1)[:3]
    return worldPos


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--transform", type=str,
                        required=True, default="Transform.h5")
    args = parser.parse_args()

    worldPos = getWorldPos(args.transform)
    print(worldPos)
