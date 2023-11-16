#pragma once

#include <atomic>
#include <vector>
#include <vtkSmartPointer.h>
#include <Eigen/Core>
#include "VelRegistrationMacro.h"

class vtkImageData;
class VelRayCastInterpolator;
class VelTwoImageToOneMetric
{
 public:
  VelTwoImageToOneMetric() = default;
  ~VelTwoImageToOneMetric() = default;

  VelSetMacro(BlockSize, int);
  VelSetPointerMacro(FixedImage1, vtkImageData);
  VelSetPointerMacro(FixedImage2, vtkImageData);
  VelSetPointerMacro(FixedImageMask1, vtkImageData);
  VelSetPointerMacro(FixedImageMask2, vtkImageData);
  VelSetPointerMacro(Interpolator1, VelRayCastInterpolator);
  VelSetPointerMacro(Interpolator2, VelRayCastInterpolator);
  /**
   * @brief 优化算法的目标函数
   *
   * @param x 待优化的参数: rx, ry, rz, tx, ty, tz
   * @return double
   */
  double operator()(Eigen::VectorXd& x);

  void Update();

  int fcalls = 0;  // 用于优化算法中统计函数调用次数

 protected:
  vtkSmartPointer<vtkImageData> m_FixedImage1;
  vtkSmartPointer<vtkImageData> m_FixedImage2;
  vtkSmartPointer<vtkImageData> m_FixedImageMask1;
  vtkSmartPointer<vtkImageData> m_FixedImageMask2;
  VelRayCastInterpolator* m_Interpolator1 = nullptr;
  VelRayCastInterpolator* m_Interpolator2 = nullptr;
  int m_BlockSize = 1024;

  // ********************************
  // 用于计算NormalizedCrossCorrelation
  double measure1, measure2;
  std::atomic<double> sff1, smm1, sfm1, sf1, sm1;
  std::atomic<double> sff2, smm2, sfm2, sf2, sm2;
  double denom1, denom2;
  // ********************************

  int numBlock1, numBlock2;
  unsigned char *fixedPtr1, *fixedPtr2;
  unsigned char *fixedMaskPtr1, *fixedMaskPtr2;
  std::vector<Eigen::Vector3d> imagePoints1, imagePoints2;
  std::vector<int> imageIndices1, imageIndices2;

  void process1(int imin, int imax);
  void process2(int imin, int imax);

 private:
  VelTwoImageToOneMetric(const VelTwoImageToOneMetric&) = delete;
  void operator=(const VelTwoImageToOneMetric&) = delete;
};
