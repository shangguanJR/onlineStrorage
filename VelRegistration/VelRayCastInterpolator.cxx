#include "VelRayCastInterpolator.h"

#include <vector>
#include <thread>

#include <vtkImageCast.h>
#include <vtkeigen/eigen/LU>
#include <vtkeigen/eigen/SVD>

VelRayCastInterpolator::VelRayCastInterpolator() {}

void VelRayCastInterpolator::SetBlockSize(int size)
{
  this->blockSize = size;
}

void VelRayCastInterpolator::SetSize(int size[3])
{
  memcpy(this->drrSize, size, 3 * sizeof(double));
}

void VelRayCastInterpolator::SetThreshold(double threshold)
{
  this->threshold = threshold;
}

void VelRayCastInterpolator::SetCT2World(const Eigen::Matrix4d& matrix)
{
  this->ct2world = matrix;
  this->world2cT = this->ct2world.inverse();
}

void VelRayCastInterpolator::SetRASToIJKMatrix(const Eigen::Matrix4d& matrix)
{
  this->RASToIJK = matrix;
  this->IJKToRAS = this->RASToIJK.inverse();
}

void VelRayCastInterpolator::SetProjectMatrix(const Eigen::MatrixXd& matrix)
{
  this->projectMatrix = matrix;
  this->projectMatrixInv = matrix.transpose() * (matrix * matrix.transpose()).inverse();
  Eigen::Matrix3d K, R;
  Eigen::Vector3d t;
  this->decomposeProjectionMatrix(matrix, K, R, t);
  this->cameraWorld << -R.inverse() * t, 1;  // P = KR[I|-C] = K[R|t], t = -RC, C = - R^-1 @ t
}

void VelRayCastInterpolator::SetMovingImage(vtkImageData* image, double spacing[3], double bounds[6])
{
  image->GetDimensions(ctSize);
  ctPointer = static_cast<short*>(image->GetScalarPointer());
  memcpy(ctBounds, bounds, 6 * sizeof(double));
  memcpy(ctSpacing, spacing, 3 * sizeof(double));
}

vtkSmartPointer<vtkImageData> VelRayCastInterpolator::GetOutput()
{
  double range[2];
  drrImage->GetScalarRange(range);
  size_t size = drrSize[0] * drrSize[1] * drrSize[2];
  for (size_t i = 0; i < size; i++)
  {
    drrPointer[i] = static_cast<short>(255.0 * (static_cast<double>(drrPointer[i]) - range[0]) / (range[1] - range[0]));
  }
  vtkNew<vtkImageCast> castFilter;
  castFilter->SetInputData(drrImage);
  castFilter->SetOutputScalarTypeToUnsignedChar();
  castFilter->Update();
  return castFilter->GetOutput();
}

void VelRayCastInterpolator::Update()
{
  // 第一次计算时, 设置一些初始参数, 避免重复计算
  if (!drrImage) this->initialize();

  // 优化算法迭代时会更新world2ct,因此相机坐标也要更新
  this->cameraCT = this->world2cT * cameraWorld;

  std::vector<std::thread> pool;
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
    {
      int extent[4] = {i * blockSize, (i + 1) * blockSize, j * blockSize, (j + 1) * blockSize};
      // this->process(extent[0], extent[1], extent[2], extent[3]);
      // 线程在std::thread()构造函数完成后就开始执行了
      pool.push_back(std::thread(&VelRayCastInterpolator::process, this, extent[0], extent[1], extent[2], extent[3]));
    }

  // 等待所有线程执行完毕
  for (int i = 0; i < row * col; i++)
  {
    pool[i].join();
  }
}

void VelRayCastInterpolator::decomposeProjectionMatrix(const Eigen::MatrixXd& P, Eigen::Matrix3d& K, Eigen::Matrix3d& R,
                                                       Eigen::Vector3d& t)
{
  Eigen::Matrix3d H, H_INV;
  Eigen::Vector3d h;
  H << P(0, 0), P(0, 1), P(0, 2), P(1, 0), P(1, 1), P(1, 2), P(2, 0), P(2, 1), P(2, 2);
  h << P(0, 3), P(1, 3), P(2, 3);

  /*
    ! P = KR[I|-C] = K[R|t], t = -RC
    ! 其中K为内参矩阵, [R|t] 为外参矩阵
    ! C为世界坐标的点坐标与相机坐标原点在世界坐标系下的坐标的平移向量
    ! [R|t]代表先旋转坐标系再进行平移, R[I|-C]代表先平移再旋转
    ! OpenCV的接口得到的posVect(transVect)是C而不是t
    P = K[R|t] = [H|h] -> H = KR, h = Kt ->
    H^-1 = (KR)^-1 = R^-1 K^-1 = R^T K^-1 = qr
    R = q^T, K = r^-1, t = K^-1 h
  */
  H_INV = H.inverse();
  Eigen::HouseholderQR<Eigen::MatrixXd> QR(H_INV);
  Eigen::MatrixXd r = QR.matrixQR().triangularView<Eigen::Upper>();
  Eigen::MatrixXd q = QR.householderQ();
  K = r.inverse();
  R = q.transpose();
  /*
    K按照定义应该为齐次矩阵, 但此处为了计算正确的R和t暂时不进行归一化
    在计算相机坐标->图像坐标的转换时，将齐次坐标进行齐次归一化一样可以得到正确结果
    // K /= K(2, 2);
  */

  Eigen::Matrix3d R_TEMP = Eigen::Matrix3d::Identity();
  // ! K(0, 0)与K(1, 1)分别为fx,fy，应该是正数
  // ! H = K*R_TEMP * R_TEMP * R, R_TEMP*R_TEMP = I
  if (K(0, 0) < 0)
  {
    if (K(1, 1) < 0)
    {
      // 绕Z轴旋转180
      R_TEMP(0, 0) = R_TEMP(1, 1) = -1;
    }
    else
    {
      // 绕Y轴旋转180
      R_TEMP(0, 0) = R_TEMP(2, 2) = -1;
    }
  }
  else if (K(1, 1) < 0)
  {
    // ??? for some reason, we never get here ??? (opencv中的注释)
    // 绕X轴旋转180
    R_TEMP(1, 1) = R_TEMP(2, 2) = -1;
  }
  K = K * R_TEMP;
  R = R_TEMP * R;

  t = K.inverse() * h;
}

void VelRayCastInterpolator::initialize()
{
  drrImage = vtkSmartPointer<vtkImageData>::New();
  drrImage->SetDimensions(drrSize);
  drrImage->AllocateScalars(VTK_SHORT, 1);
  drrPointer = static_cast<short*>(drrImage->GetScalarPointer());

  row = drrSize[0] / blockSize;
  col = drrSize[1] / blockSize;
}

void VelRayCastInterpolator::process(int imin, int imax, int jmin, int jmax)
{
  Eigen::Vector3d point;  // DRR的图像坐标
  for (int j = jmin; j < jmax; j++)
    for (int i = imin; i < imax; i++)
    {
      point << i, j, 1;
      drrPointer[i + drrSize[0] * j] = this->evaluate(point);
    }
}

short VelRayCastInterpolator::evaluate(const Eigen::Vector3d& point)
{
  short pixel;
  int ctIndex[3];  // IJK index

  Eigen::Vector4d firstIntersection = Eigen::Vector4d::Ones();
  float alphaX1, alphaXN, alphaXmin, alphaXmax;
  float alphaY1, alphaYN, alphaYmin, alphaYmax;
  float alphaZ1, alphaZN, alphaZmin, alphaZmax;
  float alphaMin, alphaMax;
  float alphaX, alphaY, alphaZ, alphaCmin, alphaCminPrev;
  float alphaUx, alphaUy, alphaUz;
  float alphaIntersectionUp[3], alphaIntersectionDown[3];
  float d12, value;
  Eigen::Vector4d firstIntersectionIndex;
  Eigen::Vector4d firstIntersectionIndexUp, firstIntersectionIndexDown;
  int iU, jU, kU;

  const short minOutputValue = VTK_SHORT_MIN;
  const short maxOutputValue = VTK_SHORT_MAX;

  // 计算经过该点的光线
  Eigen::Vector4d pointWorld = projectMatrixInv * point;
  pointWorld /= pointWorld[3];
  Eigen::Vector4d pointCT = world2cT * pointWorld;
  float rayVector[3];
  rayVector[0] = static_cast<float>(pointCT[0] - cameraCT[0]);
  rayVector[1] = static_cast<float>(pointCT[1] - cameraCT[1]);
  rayVector[2] = static_cast<float>(pointCT[2] - cameraCT[2]);

  /* Calculate the parametric  values of the first  and  the  last
  intersection points of  the  ray  with the X,  Y, and Z-planes  that
  define  the  CT volume. */
  if (rayVector[0] != 0)
  {
    alphaX1 = (ctBounds[0] - cameraCT[0]) / rayVector[0];
    alphaXN = (ctBounds[1] - cameraCT[0]) / rayVector[0];
    alphaXmin = std::min(alphaX1, alphaXN);
    alphaXmax = std::max(alphaX1, alphaXN);
  }
  else
  {
    alphaXmin = -2;
    alphaXmax = 2;
  }
  if (rayVector[1] != 0)
  {
    alphaY1 = (ctBounds[2] - cameraCT[1]) / rayVector[1];
    alphaYN = (ctBounds[3] - cameraCT[1]) / rayVector[1];
    alphaYmin = std::min(alphaY1, alphaYN);
    alphaYmax = std::max(alphaY1, alphaYN);
  }
  else
  {
    alphaYmin = -2;
    alphaYmax = 2;
  }
  if (rayVector[2] != 0)
  {
    alphaZ1 = (ctBounds[4] - cameraCT[2]) / rayVector[2];
    alphaZN = (ctBounds[5] - cameraCT[2]) / rayVector[2];
    alphaZmin = std::min(alphaZ1, alphaZN);
    alphaZmax = std::max(alphaZ1, alphaZN);
  }
  else
  {
    alphaZmin = -2;
    alphaZmax = 2;
  }

  /* Get the very first and the last alpha values when the ray
  intersects with the CT volume. */
  alphaMin = std::max(std::max(alphaXmin, alphaYmin), alphaZmin);
  alphaMax = std::min(std::min(alphaXmax, alphaYmax), alphaZmax);

  /* Calculate the parametric values of the first intersection point
  of the ray with the X, Y, and Z-planes after the ray entered the
  CT volume. */
  firstIntersection[0] = cameraCT[0] + alphaMin * rayVector[0];  // RAS坐标系下
  firstIntersection[1] = cameraCT[1] + alphaMin * rayVector[1];  // RAS坐标系下
  firstIntersection[2] = cameraCT[2] + alphaMin * rayVector[2];  // RAS坐标系下

  /* Transform LPS coordinate to the continuous index of the CT volume*/
  firstIntersectionIndex = RASToIJK * firstIntersection;

  firstIntersectionIndexUp[0] = ceil(firstIntersectionIndex[0]);
  firstIntersectionIndexUp[1] = ceil(firstIntersectionIndex[1]);
  firstIntersectionIndexUp[2] = ceil(firstIntersectionIndex[2]);
  firstIntersectionIndexUp[3] = 1.0;

  firstIntersectionIndexDown[0] = floor(firstIntersectionIndex[0]);
  firstIntersectionIndexDown[1] = floor(firstIntersectionIndex[1]);
  firstIntersectionIndexDown[2] = floor(firstIntersectionIndex[2]);
  firstIntersectionIndexDown[3] = 1.0;

  Eigen::Vector4d numeratorUp = IJKToRAS * firstIntersectionIndexUp - cameraCT;
  Eigen::Vector4d numeratorDown = IJKToRAS * firstIntersectionIndexDown - cameraCT;

  if (rayVector[0] == 0)
  {
    alphaX = 2;
    alphaUx = 999;
  }
  else
  {
    alphaIntersectionUp[0] = numeratorUp[0] / rayVector[0];
    alphaIntersectionDown[0] = numeratorDown[0] / rayVector[0];
    alphaX = std::max(alphaIntersectionUp[0], alphaIntersectionDown[0]);
    alphaUx = ctSpacing[0] / std::abs(rayVector[0]);
  }

  if (rayVector[1] == 0)
  {
    alphaY = 2;
    alphaUy = 999;
  }
  else
  {
    alphaIntersectionUp[1] = numeratorUp[1] / rayVector[1];
    alphaIntersectionDown[1] = numeratorDown[1] / rayVector[1];
    alphaY = std::max(alphaIntersectionUp[1], alphaIntersectionDown[1]);
    alphaUy = ctSpacing[1] / std::abs(rayVector[1]);
  }

  if (rayVector[2] == 0)
  {
    alphaZ = 2;
    alphaUz = 999;
  }
  else
  {
    alphaIntersectionUp[2] = numeratorUp[2] / rayVector[2];
    alphaIntersectionDown[2] = numeratorDown[2] / rayVector[2];
    alphaZ = std::max(alphaIntersectionUp[2], alphaIntersectionDown[2]);
    alphaUz = ctSpacing[2] / std::abs(rayVector[2]);
  }

  /* Calculate voxel index incremental values along the ray path. */
  iU = pointCT(0) < cameraCT(0) ? 1 : -1;
  jU = pointCT(1) < cameraCT(1) ? 1 : -1;
  kU = pointCT(2) < cameraCT(2) ? 1 : -1;

  /* Initialize the sum of the voxel intensities along the ray path to zero. */
  d12 = 0.0;

  /* Initialize the current ray position. */
  alphaCmin = std::min(std::min(alphaX, alphaY), alphaZ);

  /* Initialize the current voxel index. */
  ctIndex[0] = firstIntersectionIndexDown[0];
  ctIndex[1] = firstIntersectionIndexDown[1];
  ctIndex[2] = firstIntersectionIndexDown[2];

  /* Check if the ray is still in the CT volume */
  while (alphaCmin < alphaMax)
  {
    /* Store the current ray position */
    alphaCminPrev = alphaCmin;

    if ((alphaX <= alphaY) && (alphaX <= alphaZ))
    {
      /* Current ray front intercepts with x-plane. Update alphaX. */
      alphaCmin = alphaX;
      ctIndex[0] += iU;
      alphaX = alphaX + alphaUx;
    }
    else if ((alphaY <= alphaX) && (alphaY <= alphaZ))
    {
      /* Current ray front intercepts with y-plane. Update alphaY. */
      alphaCmin = alphaY;
      ctIndex[1] += jU;
      alphaY = alphaY + alphaUy;
    }
    else
    {
      /* Current ray front intercepts with z-plane. Update alphaZ. */
      alphaCmin = alphaZ;
      ctIndex[2] += kU;
      alphaZ = alphaZ + alphaUz;
    }

    /* If it is a valid index, get the voxel intensity. */
    if (ctIndex[0] >= 0 && ctIndex[1] >= 0 && ctIndex[2] >= 0 && ctIndex[0] < ctSize[0] && ctIndex[1] < ctSize[1] &&
        ctIndex[2] < ctSize[2])
    {
      size_t index = ctIndex[0] + ctIndex[1] * ctSize[1] + ctIndex[2] * ctSize[0] * ctSize[1];
      value = static_cast<double>(ctPointer[index]);
      /* Ignore voxels whose intensities are below the threshold. */
      if (value > threshold)
      {
        d12 += (alphaCmin - alphaCminPrev) * (value - threshold);
      }
    }
  }

  pixel = d12 < minOutputValue ? minOutputValue : d12 > maxOutputValue ? maxOutputValue : static_cast<short>(d12);
  return pixel;
}