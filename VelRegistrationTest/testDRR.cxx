#include "VelRayCastInterpolator.h"

#include <chrono>
#include <fstream>
#include <iostream>

#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkNIFTIImageReader.h>
#include <vtkNIFTIImageWriter.h>
#include <vtkeigen/eigen/Core>
#include <vtkeigen/eigen/LU>

void computeIJKToRASMatrix(double origin[3], double spacing[3], Eigen::Matrix4d& matrix)
{
  // clang-format off
  matrix << 
    -1 * spacing[0], 0 , 0, origin[0],
    0,  -1 * spacing[1], 0, origin[1],
    0,    0,    spacing[2], origin[2],
    0,    0,             0,         1;
  // clang-format on
}

void helpParameters()
{
  // clang-format off
  std::cerr << "\n";
  std::cerr << "Usage: testDRR <options> [input]\n";
  std::cerr << "       calculates the Digitally Reconstructed Radiograph from  \n";
  std::cerr << "       a CT image using Siddon-Jacob ray-tracing algorithm based on ProjectionMatrix. \n\n";
  std::cerr << "   where <options> is one or more of the following:\n\n";
  std::cerr << "       <-h>                     Display (this) usage information\n";
  std::cerr << "       <-th float>              CT intensity threshold, below which are ignored "
               "[default: 0]\n";
  std::cerr << "       <-origin float>          Origin of the CT Volume\n";
  std::cerr << "       <-size int int>          Size of DRR in number of pixels [default: 512x512]  \n";
  std::cerr << "       <-o file>                Output image filename\n\n";
  std::cerr << "                                by  Nie Zhilin (nzl@jmed.com)\n\n";
  exit(EXIT_FAILURE);
  // clang-format on
}

int main(int argc, char* argv[])
{
  char* inputName = nullptr;
  char* outputName = "drr.nii.gz";
  double threshold = 0.;
  int size[3]{512, 512, 1};
  double origin[3] = {0, 0, 0};
  if (argc < 2) helpParameters();

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

    if ((!flag) && (strcmp(argv[1], "-size") == 0))
    {
      flag = true;
      argc--;
      argv++;
      size[0] = atoi(argv[1]);
      argc--;
      argv++;
      size[1] = atoi(argv[1]);
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

    if ((!flag) && (strcmp(argv[1], "-o") == 0))
    {
      argc--;
      argv++;
      flag = true;
      outputName = argv[1];
      argc--;
      argv++;
    }

    if ((!flag))
    {
      if (inputName == nullptr)
      {
        inputName = argv[1];
        argc--;
        argv++;
      }
      else
      {
        std::cerr << "ERROR: Can not parse argument: " << argv[1] << std::endl;
        helpParameters();
      }
    }
  }

  Eigen::MatrixXd projectionMatrix(3, 4);
  // clang-format off
  projectionMatrix<<
    2.85369797e-01, 3.59194824e-02, 8.60887562e-04, -1.98236842e+01,
    2.37509258e-04, 3.65025158e-02, 2.85583707e-01, -1.91992513e+02,
   -1.92861761e-07, 1.37210243e-04, 1.26002227e-06, -1.58372689e-01;
  // clang-format on  
  Eigen::Matrix4d ct2world = Eigen::Matrix4d::Identity();  // RAS2World

  vtkNew<vtkNIFTIImageReader> reader;
  reader->SetFileName(inputName);
  reader->Update();
  double spacing[3];
  int dims[3];
  reader->GetOutput()->GetSpacing(spacing);
  reader->GetOutput()->GetDimensions(dims);

  Eigen::Matrix4d IJKToRAS, RASToIJK;
  computeIJKToRASMatrix(origin, spacing, IJKToRAS);
  RASToIJK = IJKToRAS.inverse();

  Eigen::Vector4d point1, point2;
  point1 << -0.5, -0.5, -0.5, 1;
  point2 << dims[0] - 0.5, dims[1] - 0.5, dims[2] - 0.5, 1;
  Eigen::Vector4d bounds1 = IJKToRAS * point1;  // RAS坐标系的起点
  Eigen::Vector4d bounds2 = IJKToRAS * point2;  // RAS坐标系的终点
  // RAS坐标系下[0, 0, ]对应着RA最大的点, [size[0], size[1],] 对应RA坐标最小的点
  double bounds[6] = {bounds2(0), bounds1(0), bounds2(1), bounds1(1), bounds1(2), bounds2(2)};

  VelRayCastInterpolator interpolator;
  interpolator.SetCT2World(ct2world);
  interpolator.SetMovingImage(reader->GetOutput(), spacing, bounds);
  interpolator.SetRASToIJKMatrix(RASToIJK);
  interpolator.SetProjectMatrix(projectionMatrix);
  interpolator.SetSize(size);
  interpolator.SetThreshold(threshold);
  interpolator.Update();

  vtkNew<vtkNIFTIImageWriter> writer;
  writer->SetInputData(interpolator.GetOutput());
  writer->SetFileName(outputName);
  writer->Write();
  writer->Update();
  return 0;
}