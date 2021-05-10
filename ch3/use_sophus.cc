#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {
  // Rotation matrix with 90 degrees along Z axis
  Matrix3d R = AngleAxisd(M_PI / 2, Vector3d(0, 0, 1)).toRotationMatrix();

  // or quaternion
  Quaterniond q(R);
  
  // Sophus::SO3d can be constructed from rotation matrix
  Sophus::SO3d SO3_R(R);

  // or a from a quaternion
  Sophus::SO3d SO3_q(q);

  // They are equivalent
  cout << " SO(3) from matrix:\n" << SO3_R.matrix() << "\n";
  cout << " SO(3) from quaternion:\n" << SO3_q.matrix() << "\n";
  cout << "They are equal!" << "\n***************\n";

  // Use logarithic map to get the Lie Algebra
  Vector3d so3 = SO3_R.log();
  cout << "so3 = \n" << so3.transpose() << "\n\n";
  // hat is from vector to skew-symmetric matrix
  cout << "so3 hat=\n" << Sophus::SO3d::hat(so3) << "\n";
  // inversely from matrix to vector
  cout << "so3 hat vee=" << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose() << "\n";

  // update by perturbation model
  // this is a small update
  Vector3d update_so3(1e-4, 0, 0);
  Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
  cout << "SO3 updated = \n" << SO3_updated.matrix() << "\n";
  cout << "*********************\n";


  // Similar for SE(3)
  // translation 1 along X
  Vector3d t(1, 0, 0);
  // construct SE3 from R, t
  Sophus::SE3d SE3_Rt(R, t); 
  // or SE3 from q, t
  Sophus::SE3d SE3_qt(q, t);
  cout << "SE3 from R, t= \n" << SE3_Rt.matrix() << "\n";
  cout << "SE3 from q, t= \n" << SE3_qt.matrix() << "\n";
  
  // Lie Algebra is a 6d vector - we provide a typedef 
  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  Vector6d se3 = SE3_Rt.log();
  cout << "se3 = " << se3.transpose() << "\n";

  return 0;
}

