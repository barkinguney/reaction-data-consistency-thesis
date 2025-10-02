
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <vector>
#include <array>

struct Datum {
  double x1, x2, x3, y;
};

static const std::vector<Datum> kData = {
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

// Model: y = A * exp(B/x1) * (x2^C) * (x3^D)
inline double model(double x1, double x2, double x3, const double* p) {
  const double A = p[0], B = p[1], C = p[2], D = p[3];
  return A * std::exp(B / x1) * std::pow(x2, C) * std::pow(x3, D);
}

// Weighted residual: (y_obs - y_pred) / sigma
struct Residual {
  Residual(double x1, double x2, double x3, double y, double sigma)
      : x1(x1), x2(x2), x3(x3), y(y), sigma(sigma) {}

  template <typename T>
  bool operator()(const T* const p, T* r) const {
    const T A = p[0], B = p[1], C = p[2], D = p[3];
    const T pred = A * ceres::exp(B / T(x1)) *
                   ceres::pow(T(x2), C) * ceres::pow(T(x3), D);
    r[0] = (T(y) - pred) / T(sigma);
    return true;
  }
  double x1, x2, x3, y, sigma;
};

int main() {
  // ---- 1) Prepare data and mask for log-linear init ----
  std::vector<Datum> valid;
  valid.reserve(kData.size());
  for (const auto& d : kData) {
    if (std::isfinite(d.x1) && std::isfinite(d.x2) && std::isfinite(d.x3) &&
        d.y > 0 && d.x2 > 0 && d.x3 > 0) {
      valid.push_back(d);
    }
  }
  if (valid.size() < 4) {
    std::cerr << "Not enough valid points for initialization.\n";
    return 1;
  }

  // ---- 2) Log-linear least squares to get A0,B0,C0,D0 ----
  const int n = static_cast<int>(valid.size());
  Eigen::MatrixXd X(n, 4);
  Eigen::VectorXd b(n);
  for (int i = 0; i < n; ++i) {
    const auto& d = valid[i];
    X(i, 0) = 1.0;
    X(i, 1) = 1.0 / d.x1;
    X(i, 2) = std::log(d.x2);
    X(i, 3) = std::log(d.x3);
    b(i) = std::log(d.y);
  }
  // Solve X * beta = b (QR is stable)
  Eigen::VectorXd beta = X.colPivHouseholderQr().solve(b);

  double params[4];
  params[0] = std::exp(beta(0)); // A0
  params[1] = beta(1);           // B0
  params[2] = beta(2);           // C0
  params[3] = beta(3);           // D0

  std::cout << "Initial guess (from log-linear):\n"
            << "A0=" << params[0] << ", B0=" << params[1]
            << ", C0=" << params[2] << ", D0=" << params[3] << "\n\n";

  // ---- 3) Build Ceres problem with 5% relative noise weights ----
  ceres::Problem problem;
  for (const auto& d : kData) {
    double sigma = 0.05 * d.y;        // same as Python
    if (sigma <= 0) continue;
    ceres::CostFunction* cost =
        new ceres::AutoDiffCostFunction<Residual, 1, 4>(new Residual(d.x1, d.x2, d.x3, d.y, sigma));
    problem.AddResidualBlock(cost, nullptr, params);
  }

  // Optional: set solver to LM (Trust Region with Levenberg-Marquardt)
  ceres::Solver::Options options;
  options.minimizer_type = ceres::TRUST_REGION;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 200;   // Ceres iterations (each computes many Jacobians)
  options.function_tolerance = 1e-12;
  options.gradient_tolerance = 1e-12;
  options.parameter_tolerance = 1e-12;
  options.num_threads = 0; // let Ceres decide

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n\n";

  const double A = params[0], B = params[1], C = params[2], D = params[3];
  std::cout.setf(std::ios::fixed); std::cout.precision(6);
  std::cout << "Fit params:\nA=" << A << "\nB=" << B << "\nC=" << C << "\nD=" << D << "\n";

  // ---- 4) Covariance (approx. 1-sigma uncertainties) ----
  ceres::Covariance::Options cov_opts;
  ceres::Covariance cov(cov_opts);
  std::vector<std::pair<const double*, const double*>> blocks = {
      {params, params}
  };
  if (cov.Compute(blocks, &problem)) {
    double cov_pp[16]; // 4x4 row-major
    cov.GetCovarianceBlock(params, params, cov_pp);
    // diag are variances
    double perr[4];
    for (int i = 0; i < 4; ++i) {
      perr[i] = std::sqrt(std::max(0.0, cov_pp[i*4 + i]));
    }
    std::cout << "\nParameter 1-sigma (approx):\n";
    std::cout << "A ± " << perr[0] << "\n"
              << "B ± " << perr[1] << "\n"
              << "C ± " << perr[2] << "\n"
              << "D ± " << perr[3] << "\n";
  } else {
    std::cout << "\nCovariance computation failed.\n";
  }

  // ---- 5) Fit quality (R^2) ----
  double y_mean = 0.0;
  for (const auto& d : kData) y_mean += d.y;
  y_mean /= static_cast<double>(kData.size());

  double ss_res = 0.0, ss_tot = 0.0;
  for (const auto& d : kData) {
    const double yhat = model(d.x1, d.x2, d.x3, params);
    const double r = d.y - yhat;
    ss_res += r * r;
    const double dt = d.y - y_mean;
    ss_tot += dt * dt;
  }
  const double r2 = 1.0 - ss_res / ss_tot;
  std::cout << "\nR^2 = " << r2 << "\n";

  return 0;
}
