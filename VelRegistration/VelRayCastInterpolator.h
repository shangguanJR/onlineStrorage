#pragma once

#include <vtkImageData.h>
#include <vtkSmartPointer.h>
#include <vtkeigen/eigen/Core>

class VelRayCastInterpolator
{
 public:
  VelRayCastInterpolator();
  ~VelRayCastInterpolator() = default;

  void SetBlockSize(int size);
  void SetSize(int size[3]);
  void SetThreshold(double threshold);
  void SetMovingImage(vtkImageData* image, double spacing[3], double bounds[6]);
  void SetRASToIJKMatrix(const Eigen::Matrix4d& matrix);
  void SetCT2World(const Eigen::Matrix4d& matrix);
  void SetProjectMatrix(const Eigen::MatrixXd& matrix);

  /**
   * @brief 返回生成的DRR,该DRR像素类型为unsigned char，并经过了一次Y轴翻转
   */
  vtkSmartPointer<vtkImageData> GetOutput();

  void Update();

 protected:
  int drrSize[3]{512, 512, 1};  // DRR图像大小
  int ctSize[3];                // CT图像的Size
  double ctBounds[6];           // CT图像在RAS坐标系下的边界
  double ctSpacing[6];          // CT Spacing
  double threshold = 0;         // 低于该阈值的体素被忽略
  int blockSize = 32;           // 每个线程计算blockSize*blockSize大小的区域
  int row, col;                 // size / blockSize

  Eigen::Matrix4d RASToIJK;          // CT Volume的RASToIJKMatrix
  Eigen::Matrix4d IJKToRAS;          // CT Volume的IJKToRASMatrix
  Eigen::Matrix4d world2cT;          // 待优化的目标矩阵
  Eigen::Matrix4d ct2world;          // 待优化的目标矩阵
  Eigen::MatrixXd projectMatrix;     // 投影矩阵
  Eigen::MatrixXd projectMatrixInv;  // 投影矩阵的伪逆
  Eigen::Vector4d cameraWorld;       // 相机原点在世界坐标系下的坐标
  Eigen::Vector4d cameraCT;          // 相机原点在CT坐标系(RAS)下的坐标

  short* ctPointer = nullptr;
  short* drrPointer = nullptr;
  vtkSmartPointer<vtkImageData> drrImage;

  void decomposeProjectionMatrix(const Eigen::MatrixXd& P, Eigen::Matrix3d& K, Eigen::Matrix3d& R, Eigen::Vector3d& t);
  short evaluate(const Eigen::Vector3d& point);
  void initialize();
  void process(int imin, int imax, int jmin, int jmax);

 private:
  VelRayCastInterpolator(const VelRayCastInterpolator&) = delete;
  void operator=(const VelRayCastInterpolator&) = delete;
};