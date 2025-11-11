#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <vector>
#include <array>

struct ExperimentData {
  double T5, P5, phi, ignition_delay_time;
};

static const std::vector<ExperimentData> kData = {
  {1094, 2.13, 3, 4400},
  {1106, 2.13, 3, 2100},
  {1121, 2.13, 3, 1590},
  {1144, 2.63, 3, 1740},
  {1174, 2.53, 3, 1020},
  {1176, 2.13, 3, 857},
  {1200, 2.23, 3, 810},
  {1210, 2.23, 3, 760},
  {1221, 2.23, 3, 640},
  {1240, 2.53, 3, 605},
  {1269, 2.13, 3, 345},
  {1269, 2.13, 3, 282},
  {1276, 2.13, 3, 316},
  {1314, 2.53, 3, 282},
  {1353, 2.03, 3, 107},
  {1368, 2.43, 3, 175},
  {1419, 2.13, 3, 67},
  {1419, 2.03, 3, 80},
  {1419, 1.93, 3, 67},
  {1433, 2.03, 3, 62},
  {1442, 2.03, 3, 67},
  {1467, 2.33, 3, 75},
  {1506, 2.13, 3, 37},
  {1539, 2.43, 3, 54},
};




int main() {
    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(kData.size(), 4);
    for (size_t i = 0; i < kData.size(); ++i) {
        const auto& data = kData[i];
        A(i, 0) = 1.0;
        A(i, 1) = 1.0 / data.T5;
        A(i, 2) = std::log(data.P5);
        A(i, 3) = std::log(data.phi);
    }
    Eigen::VectorXf b = Eigen::VectorXf::Zero(kData.size());
    for (size_t i = 0; i < kData.size(); ++i) {
        b(i) = std::log(kData[i].ignition_delay_time);
    }

    Eigen::VectorXf result = A.colPivHouseholderQr().solve(b);
    std::cout << "The solution using the QR decomposition is:\n" << result << std::endl;

    result[0] = std::exp(result[0]);
    std::cout << "The fitted parameters are:\n";
    std::cout << "A: " << result[0] << "\n";
    std::cout << "B: " << result[1] << "\n";
    std::cout << "C: " << result[2] << "\n";
    std::cout << "D: " << result[3] << "\n";
}
