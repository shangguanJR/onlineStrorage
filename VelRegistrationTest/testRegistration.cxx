#include "VelRayCastInterpolator.h"
#include "VelTwoImageToOneMetric.h"

#include <chrono>
#include <iostream>

#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkNIFTIImageReader.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <openMO/powell>

void helpParameters()
{
  std::cerr << "\n";
  std::cerr << "Usage: testRegistration <options> [xray1, xray2, CT]\n";
  std::cerr << "       calculates WorldToRAS Transform Matrix from 2 XRay image \n";
  std::cerr << "   where <options> is one or more of the following:\n\n";
  std::cerr << "       <-h>                     Display (this) usage information\n";
  std::cerr << "       <-v>                     Verbose output [default: no]\n";
  std::cerr << "       <-r float float float>   Rotation of WorldToRAS,[default:0,0,0]\n";
  std::cerr << "       <-t float float float>   Translation of WorldToRAS:[default:0,0,0]\n";
  std::cerr << "       <-th float>              CT intensity threshold, below which are ignored "
               "[default: 0]\n";
  std::cerr << "       <-origin float>          Origin of the CT Volume\n";
  std::cerr << "       <-m1 path>               FixedImage Mask 1\n";
  std::cerr << "       <-m2 path>               FixedImage Mask 2\n";
  std::cerr << "                                by  Nie Zhilin (nzl@jmed.com)\n\n";
  exit(EXIT_FAILURE);
}

int main(int argc, char* argv[])
{
  char* ctFileName = nullptr;
  char* xray1FileName = nullptr;
  char* xray2FileName = nullptr;
  char* mask1FileName = nullptr;
  char* mask2FileName = nullptr;
  bool verbose = false;
  double threshold = 0;
  double origin[3]{0, 0, 0};
  double rotation[3]{0, 0, 0};
  double translation[3]{0, 0, 0};
  if (argc < 4) helpParameters();

  bool flag = false;
  while (argc > 1)
  {
    flag = false;

    if ((!flag) && (strcmp(argv[1], "-h") == 0))
    {
      argc--;
      argv++;
      flag = true;
      helpParameters();
    }

    if ((!flag) && (strcmp(argv[1], "-v") == 0))
    {
      argc--;
      argv++;
      flag = true;
      verbose = true;
    }

    if ((!flag) && (strcmp(argv[1], "-th") == 0))
    {
      argc--;
      argv++;
      flag = true;
      threshold = atof(argv[1]);
      argc--;
      argv++;
    }

    if ((!flag) && (strcmp(argv[1], "-r") == 0))
    {
      flag = true;
      argc--;
      argv++;
      rotation[0] = atof(argv[1]);
      argc--;
      argv++;
      rotation[1] = atof(argv[1]);
      argc--;
      argv++;
      rotation[2] = atof(argv[1]);
      argc--;
      argv++;
    }

    if ((!flag) && (strcmp(argv[1], "-t") == 0))
    {
      flag = true;
      argc--;
      argv++;
      translation[0] = atof(argv[1]);
      argc--;
      argv++;
      translation[1] = atof(argv[1]);
      argc--;
      argv++;
      translation[2] = atof(argv[1]);
      argc--;
      argv++;
    }

    if ((!flag) && (strcmp(argv[1], "-origin") == 0))
    {
      flag = true;
      argc--;
      argv++;
      origin[0] = atof(argv[1]);
      argc--;
      argv++;
      origin[1] = atof(argv[1]);
      argc--;
      argv++;
      origin[2] = atof(argv[1]);
      argc--;
      argv++;
    }

    if ((!flag) && (strcmp(argv[1], "-m1") == 0))
    {
      argc--;
      argv++;
      flag = true;
      mask1FileName = argv[1];
      argc--;
      argv++;
    }

    if ((!flag) && (strcmp(argv[1], "-m2") == 0))
    {
      argc--;
      argv++;
      flag = true;
      mask2FileName = argv[1];
      argc--;
      argv++;
    }

    if (!flag)
    {
      if (xray1FileName == nullptr)
      {
        xray1FileName = argv[1];
        argc--;
        argv++;
      }
      if (xray2FileName == nullptr)
      {
        xray2FileName = argv[1];
        argc--;
        argv++;
      }
      if (ctFileName == nullptr)
      {
        ctFileName = argv[1];
        argc--;
        argv++;
      }
      else
      {
        std::cerr << "ERROR: Cannot parse argument " << argv[1] << std::endl;
        helpParameters();
      }
    }
  }

  vtkNew<vtkNIFTIImageReader> reader1, reader2, reader3;
  reader1->SetFileName(ctFileName);
  reader1->Update();
  reader2->SetFileName(xray1FileName);
  reader2->Update();
  reader3->SetFileName(xray2FileName);
  reader3->Update();
  vtkImageData* ct = reader1->GetOutput();
  vtkImageData* xray1 = reader2->GetOutput();
  vtkImageData* xray2 = reader3->GetOutput();

  int ctSize[3], drrSize[3];
  double spacing[3];
  ct->GetDimensions(ctSize);
  ct->GetSpacing(spacing);
  if (xray1->GetDimensions()[0] != xray2->GetDimensions()[0] || xray1->GetDimensions()[1] != xray2->GetDimensions()[1])
  {
    std::cerr << "XRay Dimensions should be equal!" << std::endl;
    return EXIT_FAILURE;
  }

  xray1->GetDimensions(drrSize);

  vtkSmartPointer<vtkImageData> mask1, mask2;
  if (mask1FileName)
  {
    vtkNew<vtkNIFTIImageReader> reader;
    reader->SetFileName(mask1FileName);
    reader->Update();
    mask1 = reader->GetOutput();
  }
  else
  {
    mask1 = vtkSmartPointer<vtkImageData>::New();
    mask1->SetDimensions(drrSize);
    mask1->AllocateScalars(VTK_UNSIGNED_CHAR, 1);
    uint8_t* ptr = static_cast<uint8_t*>(mask1->GetScalarPointer());
    for (size_t i = 0; i < drrSize[0] * drrSize[1] * drrSize[2]; i++)
    {
      ptr[i] = 1;
    }
  }
  if (mask2FileName)
  {
    vtkNew<vtkNIFTIImageReader> reader;
    reader->SetFileName(mask2FileName);
    reader->Update();
    mask2 = reader->GetOutput();
  }
  else
  {
    mask2 = vtkSmartPointer<vtkImageData>::New();
    mask2->SetDimensions(drrSize);
    mask2->AllocateScalars(VTK_UNSIGNED_CHAR, 1);
    uint8_t* ptr = static_cast<uint8_t*>(mask2->GetScalarPointer());
    for (size_t i = 0; i < drrSize[0] * drrSize[1] * drrSize[2]; i++)
    {
      ptr[i] = 1;
    }
  }

  // TODO 两个角度的投影矩阵
  Eigen::Matrix4d P1, P2;

  VelRayCastInterpolator interpolator1;
  VelRayCastInterpolator interpolator2;
  interpolator1.SetDRRSize(drrSize);
  interpolator1.SetMovingImage(ct, spacing, origin);
  interpolator1.SetProjectMatrix(P1);
  interpolator1.SetThreshold(threshold);

  interpolator2.SetDRRSize(drrSize);
  interpolator2.SetMovingImage(ct, spacing, origin);
  interpolator2.SetProjectMatrix(P1);
  interpolator2.SetThreshold(threshold);

  VelTwoImageToOneMetric metric;
  metric.SetFixedImage1(xray1);
  metric.SetFixedImage2(xray2);
  metric.SetFixedImageMask1(mask1);
  metric.SetFixedImageMask2(mask2);
  metric.SetInterpolator1(&interpolator1);
  metric.SetInterpolator2(&interpolator2);
  metric.Update();

  const double dtr = 0.017453292519943295;
  Eigen::VectorXd x(6);
  mo::Powell<VelTwoImageToOneMetric> powell(6);
  x << rotation[0] * dtr, rotation[1] * dtr, rotation[2] * dtr, translation[0], translation[1], translation[2];
  mo::PowellResult res = powell.optimize(metric, x);
  auto finalParameters = res.x;
  Eigen::Matrix4d world2ct = interpolator1.GetWorld2RAS();
  if (verbose)
  {
    std::cout << res << std::endl;
    std::cout << "Rotation Along X:" << finalParameters(0) / dtr << "°" << std::endl;
    std::cout << "Rotation Along Y:" << finalParameters(1) / dtr << "°" << std::endl;
    std::cout << "Rotation Along Z:" << finalParameters(2) / dtr << "°" << std::endl;
    std::cout << "Translation X:" << finalParameters(3) << "mm" << std::endl;
    std::cout << "Translation Y:" << finalParameters(4) << "mm" << std::endl;
    std::cout << "Translation Z:" << finalParameters(5) << "mm" << std::endl;
    std::cout << "World2CT(RAS):\n" << world2ct << std::endl;
  }
}