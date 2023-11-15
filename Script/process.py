import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
from scipy.linalg import svd
from getMarkerPosition import getWorldPos, Tras2lps


def sort(points: np.ndarray):
    km = KMeans(n_clusters=6, random_state=2023)
    labels = km.fit_predict(points[:, 1].reshape(-1, 1))
    # 创建一个字典来存储每个聚类的点
    cluster_dict = {}
    for i in range(len(labels)):
        if labels[i] not in cluster_dict:
            cluster_dict[labels[i]] = []
        cluster_dict[labels[i]].append(points[i])

    # 对每个聚类内的点按照 x 坐标进行排序
    sorted_points = []
    for label in np.argsort(km.cluster_centers_.reshape(-1))[::-1]:
        cluster_points = np.array(cluster_dict[label])
        sorted_cluster = cluster_points[np.argsort(cluster_points[:, 0])]
        sorted_points.extend(sorted_cluster)

    # # 按照 y 坐标对6个类别进行排序
    sorted_points = np.array(sorted_points)

    return sorted_points


def normalizeTransform2D(pts):
    std = np.std(pts)
    s = std / np.sqrt(2)
    center = pts.mean(0)
    T = np.zeros((3, 3))
    T[0, 0] = T[1, 1] = 1 / s
    T[0, 2] = -1 / s * center[0]
    T[1, 2] = -1 / s * center[1]
    T[2, 2] = 1
    return T


def normalizeTransform3D(pts):
    std = np.std(pts)
    s = std / np.sqrt(3)
    center = pts.mean(0)
    T = np.zeros((4, 4))
    T[0, 0] = T[1, 1] = T[2, 2] = 1 / s
    T[0, 3] = -1 / s * center[0]
    T[1, 3] = -1 / s * center[1]
    T[2, 3] = -1 / s * center[2]
    T[3, 3] = 1
    return T


def computeProjectionMatrix(image_points, world_points):
    T_i = normalizeTransform2D(image_points)
    T_w = normalizeTransform3D(world_points)

    world_homo = np.ones((len(world_points), 4))
    image_homo = np.ones((len(image_points), 3))
    world_homo[:, :3] = world_points
    image_homo[:, :2] = image_points
    world_norm = T_w@world_homo.T
    image_norm = T_i@image_homo.T

    A = np.zeros((2*len(image_points), 12))
    for i in range(len(image_points)):
        x = world_norm[0, i]
        y = world_norm[1, i]
        z = world_norm[2, i]
        u = image_norm[0, i]
        v = image_norm[1, i]
        A[2*i] = (x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u)
        A[2*i+1] = (0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v)

    U, s, VT = svd(A)
    P_norm = VT.T[:, 11].reshape(3, 4)
    P = np.linalg.inv(T_i)@P_norm@T_w
    return P


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True)
    parser.add_argument("-t", "--transform", type=str, required=True)
    args = parser.parse_args()

    image = cv2.imread(args.file, 0)
    image[image < 200] = 0
    image[image > 0] = 1

    labeled_image = label(image, connectivity=image.ndim)
    regions = regionprops(labeled_image)

    plt.imshow(image, 'gray')
    centroids = []
    for i in range(len(regions)):
        centroid = regions[i].centroid
        centroids.append(centroid)

    centroids = np.array(centroids)
    centroids = centroids[:, ::-1]

    centroids = sort(centroids)
    for i in range(len(centroids)):
        centroid = centroids[i]
        plt.scatter(centroid[0], centroid[1])
        plt.text(centroid[0], centroid[1], str(i+1), color='w')
    plt.show()

    imagePoints = centroids
    worldPoints = getWorldPos(args.transform)
    projectMatrix = computeProjectionMatrix(imagePoints, worldPoints)
    K, R, C, _, _, _, _ = cv2.decomposeProjectionMatrix(projectMatrix)
    C /= C[3, 0]
    C = C.reshape(-1)[:3]
    cameraPosition = np.array([-73.83789062, 614.16210938, 530., 1])
    cameraPosition = Tras2lps @ cameraPosition.reshape(4, 1)
    cameraPosition /= cameraPosition[3, 0]
    cameraPosition = cameraPosition.reshape(-1)[:3]
    distance = np.linalg.norm(cameraPosition - C)
    print("CameraWorld:", C, " Distance:", distance)
    print(projectMatrix)
