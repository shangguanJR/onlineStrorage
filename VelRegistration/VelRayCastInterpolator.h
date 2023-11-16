#pragma once

#include <vtkImageData.h>
#include <vtkSmartPointer.h>
#include <Eigen/Core>
#include "VelRegistrationMacro.h"

class VelRayCastInterpolator
{
 public:
  VelRayCastInterpolator() = default;
  ~VelRayCastInterpolator() = default;

  VelSetMacro(BlockSize, int);
  VelSetMacro(Threshold, double);
  VelSetVector3Macro(DRRSize, int);
  VelSetVector3Macro(Rotation, double);
  VelSetVector3Macro(Translation, double);

  /**
   * @brief 3DSlicer中,spacing和origin都记录在Node里, 需要单独设置
   * 通过origin可以计算得到RAS2LPS的转换矩阵
   */
  void SetMovingImage(vtkImageData* image, double spacing[3], double origin[3]);

  /**
   * @brief 投影矩阵的伪逆可以得到图像坐标到世界坐标的转换, 分解投影矩阵可以得到相机原点在世界坐标中的位置
   */
  void SetProjectMatrix(const Eigen::MatrixXd& matrix);

  /**
   * @brief 返回生成的DRR,该DRR像素类型为unsigned char
   */
  vtkSmartPointer<vtkImageData> GetOutput();

  /**
   * @brief 返回目标矩阵World2RAS
   */
  Eigen::Matrix4d GetWorld2RAS() const;

  void Update();

 protected:
  int m_DRRSize[3]{512, 512, 1};  // DRR图像大小
  double m_Threshold = 0;         // 低于该阈值的体素被忽略
  int m_BlockSize = 32;           // 每个线程计算blockSize*blockSize大小的区域
  double m_Rotation[3];           // 用于计算world2RAS的旋转角度
  double m_Translation[3];        // 用于计算world2RAS的平移
  int row, col;                   // size / blockSize
  int ctSize[3];                  // CT图像的Size
  double ctSpacing[6];            // CT Spacing

  Eigen::Matrix4d RAS2LPS;           // 3DSlicer的RAS坐标与用于生成DRR的LPS坐标转换
  Eigen::Matrix4d world2RAS;         // 待优化的目标矩阵
  Eigen::MatrixXd projectMatrix;     // 投影矩阵
  Eigen::MatrixXd projectMatrixInv;  // 投影矩阵的伪逆
  Eigen::Vector4d cameraWorld;       // 相机原点在世界坐标系下的坐标
  Eigen::Vector4d cameraLPS;         // 相机原点在CT坐标系(LPS)下的坐标

  short* ctPointer = nullptr;
  short* drrPointer = nullptr;
  vtkSmartPointer<vtkImageData> drrImage;

  void computeWorld2RAS();
  void decomposeProjectionMatrix(const Eigen::MatrixXd& P, Eigen::Matrix3d& K, Eigen::Matrix3d& R, Eigen::Vector3d& t);
  short evaluate(const Eigen::Vector3d& point);
  void initialize();
  void process(int imin, int imax, int jmin, int jmax);
  void Rx(double center[3], double angle, Eigen::Matrix4d& out);
  void Ry(double center[3], double angle, Eigen::Matrix4d& out);
  void Rz(double center[3], double angle, Eigen::Matrix4d& out);

 private:
  VelRayCastInterpolator(const VelRayCastInterpolator&) = delete;
  void operator=(const VelRayCastInterpolator&) = delete;

  friend class VelTwoImageToOneMetric;
};