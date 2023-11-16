#include "VelRayCastInterpolator.h"

#include <chrono>
#include <iostream>

#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkNIFTIImageReader.h>
#include <vtkNIFTIImageWriter.h>
#include <Eigen/Core>
#include <Eigen/LU>

void helpParameters()
{
  // clang-format off
  std::cerr << "\n";
  std::cerr << "Usage: testDRR <options> [input]\n";
  std::cerr << "       calculates the Digitally Reconstructed Radiograph from  \n";
  std::cerr << "       a CT image using Siddon-Jacob ray-tracing algorithm based on ProjectionMatrix. \n\n";
  std::cerr << "   where <options> is one or more of the following:\n\n";
  std::cerr << "       <-h>                     Display (this) usage information\n";
  std::cerr << "       <-r float float float>   Rotation of WorldToRAS,[default:0,0,0]\n";
  std::cerr << "       <-t float float float>   Translation of WorldToRAS:[default:0,0,0]\n";
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
  double rotation[3]{0, 0, 0};
  double translation[3]{0, 0, 0};
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
  // 15个点
  projectionMatrix <<
    2.85369797e-01, 3.59194824e-02, 8.60887562e-04, -1.98236842e+01,
    2.37509258e-04, 3.65025158e-02, 2.85583707e-01, -1.91992513e+02,
   -1.92861761e-07, 1.37210243e-04, 1.26002227e-06, -1.58372689e-01;

  // 36个点, 影像标定器, 距离相机近
  projectionMatrix <<
    -4.95172611e-01, -1.31909820e-01, -1.07993006e-04,  4.46805453e+01,
    -7.64596168e-05, -1.31796109e-01, -4.95300098e-01,  3.43657448e+02,
    -2.23972152e-07, -5.18168255e-04, -3.78537617e-07,  3.18453779e-01;

  // // 36个点, 影像标定器, 距离相机中
  // projectionMatrix << 
  //    3.90629921e-01, 1.04616234e-01, 4.50721037e-05, -3.53675081e+01,
  //   -2.81831083e-05, 1.04865854e-01, 3.90782751e-01, -2.71347488e+02,
  //   -9.35558107e-08, 4.08845298e-04, 4.35314331e-08, -2.51144489e-01;

  // // 36个点, 影像标定器, 距离相机远
  // projectionMatrix << 
  //   -2.72690623e-01, -7.18933999e-02,  1.48433093e-05, 2.45864413e+01,
  //   -5.39990801e-05, -7.23723319e-02, -2.72539968e-01, 1.89185909e+02,
  //   -2.49669213e-07, -2.82347220e-04,  7.29677506e-08, 1.75037004e-01;
  // clang-format on

  vtkNew<vtkNIFTIImageReader> reader;
  reader->SetFileName(inputName);
  reader->Update();
  double spacing[3];
  int dims[3];
  reader->GetOutput()->GetSpacing(spacing);
  reader->GetOutput()->GetDimensions(dims);

  // ! 通过rx,ry,rz,tx,ty,tz来计算ras2world,即配准的目标矩阵,这样只需要优化6个参数
  const double dtr = 0.017453292519943295;
  rotation[0] *= dtr;
  rotation[1] *= dtr;
  rotation[2] *= dtr;
  VelRayCastInterpolator interpolator;
  interpolator.SetMovingImage(reader->GetOutput(), spacing, origin);
  interpolator.SetProjectMatrix(projectionMatrix);
  interpolator.SetRotation(rotation);
  interpolator.SetTranslation(translation);
  interpolator.SetDRRSize(size);
  interpolator.SetThreshold(threshold);
  interpolator.Update();

  vtkNew<vtkNIFTIImageWriter> writer;
  writer->SetInputData(interpolator.GetOutput());
  writer->SetFileName(outputName);
  writer->Write();
  writer->Update();
  return 0;
}