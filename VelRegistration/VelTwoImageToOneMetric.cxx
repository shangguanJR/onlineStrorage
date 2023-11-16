#include "VelTwoImageToOneMetric.h"
#include "VelRayCastInterpolator.h"

#include <vtkImageData.h>
#include <thread>

void VelTwoImageToOneMetric::Update()
{
  m_Interpolator1->initialize();
  m_Interpolator2->initialize();
  fixedPtr1 = static_cast<unsigned char*>(m_FixedImage1->GetScalarPointer());
  fixedPtr2 = static_cast<unsigned char*>(m_FixedImage2->GetScalarPointer());
  fixedMaskPtr1 = static_cast<unsigned char*>(m_FixedImageMask1->GetScalarPointer());
  fixedMaskPtr2 = static_cast<unsigned char*>(m_FixedImageMask2->GetScalarPointer());

  imageIndices1.clear();
  imageIndices2.clear();
  imagePoints1.clear();
  imagePoints2.clear();
  
  int dims[3];
  m_FixedImage1->GetDimensions(dims);
  for (int j = 0; j < dims[1]; j++)
  {
    for (int i = 0; i < dims[0]; j++)
    {
      int index = i + j * dims[0];
      if (fixedMaskPtr1[index])
      {
        Eigen::Vector3d point(i, j, 1.0);
        imagePoints1.push_back(point);
        imageIndices1.push_back(index);
      }
      if (fixedMaskPtr2[index])
      {
        Eigen::Vector3d point(i, j, 1.0);
        imagePoints2.push_back(point);
        imageIndices2.push_back(index);
      }
    }
  }

  numBlock1 = imageIndices1.size() / m_BlockSize;
  numBlock2 = imageIndices2.size() / m_BlockSize;
}

double VelTwoImageToOneMetric::operator()(Eigen::VectorXd& x)
{
  fcalls++;
  double r[3] = {x[0], x[1], x[2]};
  double t[3] = {x[3], x[4], x[5]};

  m_Interpolator1->SetRotation(r);
  m_Interpolator1->SetTranslation(t);
  m_Interpolator1->computeWorld2RAS();

  m_Interpolator1->SetRotation(r);
  m_Interpolator1->SetTranslation(t);
  m_Interpolator1->computeWorld2RAS();

  sff1 = smm1 = sfm1 = sf1 = sm1 = 0;
  sff2 = smm2 = sfm2 = sf2 = sm2 = 0;
  std::vector<std::thread> pool;
  for (int i = 0; i < numBlock1; i++)
  {
    pool.push_back(std::thread(&VelTwoImageToOneMetric::process1, this, i * m_BlockSize, (i + 1) * m_BlockSize));
  }
  if (imageIndices1.size() % m_BlockSize != 0)
  {
    pool.push_back(std::thread(&VelTwoImageToOneMetric::process1, this, numBlock1 * m_BlockSize, imageIndices1.size()));
  }

  for (int i = 0; i < numBlock2; i++)
  {
    pool.push_back(std::thread(&VelTwoImageToOneMetric::process2, this, i * m_BlockSize, (i + 1) * m_BlockSize));
  }
  if (imageIndices2.size() % m_BlockSize != 0)
  {
    pool.push_back(std::thread(&VelTwoImageToOneMetric::process2, this, numBlock2 * m_BlockSize, imageIndices2.size()));
  }

  double sff, smm, sfm;
  sff = sff1.load() - (sf1.load() * sf1.load()) / imageIndices1.size();
  smm = smm1.load() - (sm1.load() * sm1.load()) / imageIndices1.size();
  sfm = sfm1.load() - (sf1.load() * sm1.load()) / imageIndices1.size();
  denom1 = -1.0 * sqrt(sff * smm);
  measure1 = denom1 != 0.0 ? sfm / denom1 : 0;

  sff = sff2.load() - (sf2.load() * sf2.load()) / imageIndices2.size();
  smm = smm2.load() - (sm2.load() * sm2.load()) / imageIndices2.size();
  sfm = sfm2.load() - (sf2.load() * sm2.load()) / imageIndices2.size();
  denom2 = -1.0 * sqrt(sff * smm);
  measure2 = denom2 != 0.0 ? sfm / denom2 : 0;
  return (measure1 + measure2) * 0.5;
}

void VelTwoImageToOneMetric::process1(int imin, int imax)
{
  double fixedValue, movingValue, sff, sfm, smm, sf, sm, oldValue;
  sff = sfm = smm = sf = sm = 0;
  for (int i = imin; i < imax; i++)
  {
    movingValue = static_cast<double>(m_Interpolator1->evaluate(imagePoints1[i]));
    fixedValue = static_cast<double>(fixedPtr1[imageIndices1[i]]);
    sff += fixedValue * fixedValue;
    sfm += fixedValue * movingValue;
    smm += movingValue * movingValue;
    sf += fixedValue;
    sm += movingValue;
  }

  // 对double类型进行原子操作, 避免DataRace.效率高于互斥锁
  // https://preshing.com/20150402/you-can-do-any-kind-of-atomic-read-modify-write-operation/
  oldValue = sff1.load();
  while (!sff1.compare_exchange_weak(oldValue, oldValue + sff))
  {
  }
  oldValue = sfm1.load();
  while (!sfm1.compare_exchange_weak(oldValue, oldValue + sfm))
  {
  }
  oldValue = smm1.load();
  while (!smm1.compare_exchange_weak(oldValue, oldValue + smm))
  {
  }
  oldValue = sf1.load();
  while (!sf1.compare_exchange_weak(oldValue, oldValue + sf))
  {
  }
  oldValue = sm1.load();
  while (!sm1.compare_exchange_weak(oldValue, oldValue + sm))
  {
  }
}

void VelTwoImageToOneMetric::process2(int imin, int imax)
{
  double fixedValue, movingValue, sff, sfm, smm, sf, sm, oldValue;
  sff = sfm = smm = sf = sm = 0;
  for (int i = imin; i < imax; i++)
  {
    movingValue = static_cast<double>(m_Interpolator2->evaluate(imagePoints2[i]));
    fixedValue = static_cast<double>(fixedPtr2[imageIndices2[i]]);
    sff += fixedValue * fixedValue;
    sfm += fixedValue * movingValue;
    smm += movingValue * movingValue;
    sf += fixedValue;
    sm += movingValue;
  }

  // 对double类型进行原子操作, 避免DataRace.效率高于互斥锁
  // https://preshing.com/20150402/you-can-do-any-kind-of-atomic-read-modify-write-operation/
  oldValue = sff2.load();
  while (!sff2.compare_exchange_weak(oldValue, oldValue + sff))
  {
  }
  oldValue = sfm2.load();
  while (!sfm2.compare_exchange_weak(oldValue, oldValue + sfm))
  {
  }
  oldValue = smm2.load();
  while (!smm2.compare_exchange_weak(oldValue, oldValue + smm))
  {
  }
  oldValue = sf2.load();
  while (!sf2.compare_exchange_weak(oldValue, oldValue + sf))
  {
  }
  oldValue = sm2.load();
  while (!sm2.compare_exchange_weak(oldValue, oldValue + sm))
  {
  }
}