#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <fstream>
#include <iomanip>
#include <locale>
#include <string>
#include <filesystem>
#include <nlohmann/json.hpp> 
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include "../libs/alglib-cpp/src/interpolation.h"


using namespace alglib;

using json = nlohmann::json;

class UncertaintyQuantification {
public:
    struct FixedIFs {
        double driver_length_m;             // Driver length [m]
        double driven_length_m;             // Driven length [m]
        double driven_inner_diameter_cm;    // Driven inner diameter [cm]
        double measurement_location_cm;     // Measurement location [cm]
        bool tailoring;                     // Tailoring
        bool inserts;                       // Inserts
        bool crv;                           // CRV
        bool dilution;                      // Dilution
        bool impurities;                    // Impurities
        bool fuel_air_ratio;                // Fuel-air ratio exceeding 0.3
    };

    struct TestTimeIFs {
        double IDT;         // Ignition delay time[μs]
        double T5;          // Temperature [K]
        double P5;          // Pressure [atm]
    };

    struct FitParams {
        double A, B, C, D;     // regression coefficients
        double UA, UB, UC, UD; // uncertainties of regression coefficients
    };

    struct Experiment {
        FixedIFs fixed_IFs;    // Fixed IFs for the experiment
        std::vector<TestTimeIFs> test_time_IF_list;  // Test time IFs for the experiment
        double phi;            // Stoichiometric ratio
        double uT5;           // Uncertainty (fraction) of T5
        double uP5;           // Uncertainty (fraction) of P5
        double uphi;          // Uncertainty (fraction) of phi
        std::string mixture;   // Mixture description
    };

    struct Results {
        std::vector<double> systematic_uncertainty;
        std::vector<double> random_uncertainty;
        std::vector<double> combined_uncertainty;
        std::vector<double> final_extended_uncertainty;
        std::vector<double> percent_uncertainty;
    };

    struct IgnitionResidual {
        IgnitionResidual(double T, double P, double phi, double tau, double Tref, double Pref, double sigma = 1.0)
            : T(T), P(P), phi(phi), tau(tau), Tref(Tref), Pref(Pref), sigma(sigma) {}

        template <typename Ttype>
        bool operator()(const Ttype* const params, Ttype* residual) const {
            Ttype Aref = params[0];
            Ttype B    = params[1];
            Ttype C    = params[2];

            Ttype model = Aref * ceres::exp(B * (Ttype(1.0)/T - Ttype(1.0)/Tref)) 
                        * ceres::pow(Ttype(P)/Pref, C) 
                        * Ttype(phi);

            residual[0] = (model - Ttype(tau)) / Ttype(sigma); // weighted residual
            return true;
        }

        double T, P, phi, tau, Tref, Pref, sigma;
    };

    UncertaintyQuantification(const std::string& filepath) 
        : experiment(readJSON(filepath)) {}

    static Results do_uncertainty_quantification(const Experiment& exp) {
        Results out;

        out.systematic_uncertainty =
            calc_systematic_uncertainties(mu_ideal_fixed_IFs_bounds_, 
                                          exp.fixed_IFs,
                                          exp.test_time_IF_list,
                                          mu_ideal_bounds_test_time_IF_);

        out.random_uncertainty =
            calc_random_uncertainties(exp.test_time_IF_list, exp.phi, exp.uT5, exp.uP5, exp.uphi);

        auto [combined, final_ext, percent] =
            calc_final_extended_uncertainties(exp.test_time_IF_list,
                                              out.systematic_uncertainty,
                                              out.random_uncertainty);

        out.combined_uncertainty = std::move(combined);
        out.final_extended_uncertainty = std::move(final_ext);
        out.percent_uncertainty = std::move(percent);

        return out;
    }

    bool writeCSV(const std::string& filepath) const {
        auto res = do_uncertainty_quantification(experiment);
        const auto& tests = experiment.test_time_IF_list;  

        if (tests.size() != res.percent_uncertainty.size()) {
            std::cerr << "Mismatch in test size and results size\n";   
            return false;
        }
        std::ofstream out(filepath);
        if (!out){
            std::cerr << "Could not open " << filepath << " for writing.\n";
            return false;
        }

        out.imbue(std::locale::classic());
        out << std::fixed << std::setprecision(2);

        out << "ignition_delay_time,T5_K,P5_atm,"
            "systematic,random,combined,final_extended,percent\n";

        for (size_t i = 0; i < tests.size(); ++i) {
            out << tests[i].IDT << ','
                << tests[i].T5        << ','
                << tests[i].P5        << ','
                << res.systematic_uncertainty[i]      << ','
                << res.random_uncertainty[i]          << ','
                << res.combined_uncertainty[i]        << ','
                << res.final_extended_uncertainty[i]  << ','
                << res.percent_uncertainty[i]
                << '\n';
        }
        return true;
    }

   
private:
    inline static double delta_0_ = 5.0;

    inline static std::vector<FixedIFs> mu_ideal_fixed_IFs_bounds_ = {
        {3.0, 8.0, 10.0, 0.0, true, true, true, true, false, true},
        {std::numeric_limits<double>::infinity(), 
            std::numeric_limits<double>::infinity(), 
            std::numeric_limits<double>::infinity(), 
            2.0, 
            true, true, true, true, false, true}
    };
    inline static std::vector<TestTimeIFs> mu_ideal_bounds_test_time_IF_ = {
            {50.0, 1000.0, 5.0},
            {500.0, 1600.0, 20.0}
        };
    Experiment experiment;

    static Experiment readJSON(const std::string& filepath) {
        
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Could not open " << filepath << "\n";
            return {};
        }

        json j;
        file >> j;
    
        Experiment experiment;
        experiment.phi = j["Experiment"]["phi"];
        experiment.mixture = j["Experiment"]["mixture"];

        experiment.fixed_IFs.driver_length_m = j["FixedIFs"]["driver_length_m"];
        experiment.fixed_IFs.driven_length_m = j["FixedIFs"]["driven_length_m"];
        experiment.fixed_IFs.driven_inner_diameter_cm = j["FixedIFs"]["driven_inner_diameter_cm"];
        experiment.fixed_IFs.tailoring = j["FixedIFs"]["tailoring"];
        experiment.fixed_IFs.inserts = j["FixedIFs"]["inserts"];
        experiment.fixed_IFs.crv = j["FixedIFs"]["crv"];
        experiment.fixed_IFs.dilution = j["FixedIFs"]["dilution"];
        experiment.fixed_IFs.impurities = j["FixedIFs"]["impurities"];
        experiment.fixed_IFs.measurement_location_cm = j["FixedIFs"]["measurement_location_cm"];
        experiment.fixed_IFs.fuel_air_ratio = experiment.phi > 0.3;

        experiment.uT5 = j["Experiment"]["T5_uncertainty"];
        experiment.uP5 = j["Experiment"]["P5_uncertainty"];
        experiment.uphi = j["Experiment"]["phi_uncertainty"];

        for (const auto& entry : j["Experiment"]["TestTimeIFsList"]) {
            TestTimeIFs test_time_IF;
            test_time_IF.IDT = entry["IDT"];
            test_time_IF.T5 = entry["T5"];
            test_time_IF.P5 = entry["P5"];
            experiment.test_time_IF_list.push_back(test_time_IF);
        }

        

        return experiment;

    }

    static double harrington_desirability(double lower, double upper, double mu_actual) {
        double mu_ideal;
        if (mu_actual < lower) {
            mu_ideal = lower;
        } else if (mu_actual > upper) {
            mu_ideal = upper;
        } else {
            return 0.0; // within bounds, no penalty
        }
        double x_i = (mu_ideal - mu_actual) / mu_ideal;
        double xi = 1.0 - std::exp(-(x_i * x_i));
        double delta_i = delta_0_ * xi;
        return delta_i;
    }

    static double harrington_desirability_bool(bool mu_ideal, bool mu_actual){
        double binary_IF_penalty = 0.37;  // Penalty for boolean IFs
        double xi = (mu_ideal == mu_actual) ? 0.0 : binary_IF_penalty;
        double delta_i = delta_0_ * xi;
        return delta_i;
    }

    static double calc_total_fixed_harrington(const std::vector<FixedIFs>& mu_ideal_bounds, const FixedIFs& mu_actual) {
        double total = 0.0;

        total += harrington_desirability(mu_ideal_bounds[0].driver_length_m, mu_ideal_bounds[1].driver_length_m, mu_actual.driver_length_m);
        total += harrington_desirability(mu_ideal_bounds[0].driven_length_m,           mu_ideal_bounds[1].driven_length_m, mu_actual.driven_length_m);
        total += harrington_desirability(mu_ideal_bounds[0].driven_inner_diameter_cm,  mu_ideal_bounds[1].driven_inner_diameter_cm, mu_actual.driven_inner_diameter_cm);
        total += harrington_desirability(mu_ideal_bounds[0].measurement_location_cm,   mu_ideal_bounds[1].measurement_location_cm, mu_actual.measurement_location_cm);
        total += harrington_desirability_bool(mu_ideal_bounds[0].tailoring,            mu_actual.tailoring);
        total += harrington_desirability_bool(mu_ideal_bounds[0].inserts,              mu_actual.inserts);
        total += harrington_desirability_bool(mu_ideal_bounds[0].crv,                  mu_actual.crv);
        total += harrington_desirability_bool(mu_ideal_bounds[0].dilution,             mu_actual.dilution);
        total += harrington_desirability_bool(mu_ideal_bounds[0].impurities,           mu_actual.impurities);
        total += harrington_desirability_bool(mu_ideal_bounds[0].fuel_air_ratio,       mu_actual.fuel_air_ratio);
        return total;
    }

    static std::vector<double> calc_systematic_uncertainties(const std::vector<FixedIFs>& fixed_IF_bounds, 
                                                      const FixedIFs& actual_fixed,
                                                      const std::vector<TestTimeIFs>& test_list,
                                                      const std::vector<TestTimeIFs>& ideal_bounds){
        double fixed_IF_error = calc_total_fixed_harrington(fixed_IF_bounds, actual_fixed);

        std::vector<double> systematic_uncertainties;
        systematic_uncertainties.reserve(test_list.size());

        //Only use FixedIFs and IDT for systematic uncertainty. T5,P5 not necessary 
        double lower_bound = ideal_bounds.front().IDT;
        double upper_bound = ideal_bounds.back().IDT;

        for (const auto& entry : test_list) {
            double mu_actual = entry.IDT;
            double delta_val = 0.0;

            delta_val = harrington_desirability(lower_bound, upper_bound, mu_actual);
            double systematic_error = delta_val + fixed_IF_error + delta_0_;
            double systematic_uncertainty = std::pow((entry.IDT * systematic_error / 100.0 / 2.0), 2);
            systematic_uncertainties.push_back(systematic_uncertainty);
        }

        return systematic_uncertainties;
    }


    static FitParams least_squares_variance_wls_no_D(const std::vector<TestTimeIFs>& test_list,
                                    double phi,
                                    double frac_sigma = 0.03){
            // Build design matrix on log(IDT), same as your code
            const int n = static_cast<int>(test_list.size());
            const int p = 3;

            Eigen::MatrixXd A(n, p);
            Eigen::VectorXd b(n);
            for (int i = 0; i < n; ++i) {
                A(i, 0) = 1.0;
                A(i, 1) = 1.0 / test_list[i].T5;
                A(i, 2) = std::log(test_list[i].P5);
                b(i)    = std::log(test_list[i].IDT / phi);
            }

            
            // --- WLS with constant sigma in log-space ---
            // sigma_log = frac_sigma  ==> w = 1 / sigma^2 = 1 / (frac_sigma^2)
            const double sigma_log = std::max(frac_sigma, 1e-12);
            const double w_scalar  = 1.0 / (sigma_log * sigma_log);
            const double ws        = std::sqrt(w_scalar); // row scale

            // Pre-whiten (constant row scale)
            Eigen::MatrixXd Aw = A * ws;
            Eigen::VectorXd bw = b * ws;

            // Solve (robust SVD, same as your code)
            Eigen::VectorXd x = Aw.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(bw);

            // Residuals (in original log-space)
            Eigen::VectorXd y_pred = A * x;
            Eigen::VectorXd r = b - y_pred;

            // With absolute sigma supplied (like scipy absolute_sigma=True):
            // s^2 = 1, covariance scales purely by (Aᵀ W A)^(-1)
            const double s2 = 1.0;
        
            
            // Covariance via SVD of Aw (AwᵀAw = Aᵀ W A)
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(Aw, Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::VectorXd s = svd.singularValues();
            Eigen::MatrixXd V = svd.matrixV();

            Eigen::VectorXd inv_s2 = s.array().inverse().square();
            for (int i = 0; i < inv_s2.size(); ++i)
                if (!std::isfinite(inv_s2[i]) || s[i] < 1e-12) inv_s2[i] = 0.0;

            // (Aᵀ W A)^+ = V * diag(1/s^2) * Vᵀ
            Eigen::MatrixXd XtWX_pinv = V * inv_s2.asDiagonal() * V.transpose();
            Eigen::MatrixXd cov_beta  = s2 * XtWX_pinv;
            Eigen::VectorXd se        = cov_beta.diagonal().array().sqrt();

            // Back-transform A from log-space and propagate its SE
            x[0]  = std::exp(x[0]);
            double se_logA = se[0];
            double UA      = x[0] * se_logA;

            FitParams params;
            params.A  = x[0];
            params.B  = x[1];
            params.C  = x[2];
            // propagate SE for A with first-order approx σ_A = (dA/dβ0) * σ_β0, with ​​A = exp(β0)
            params.UA = UA;
            params.UB = se[1];
            params.UC = se[2];
        

            std::cout << "Fitted parameters: A=" << params.A
                    << ", B=" << params.B
                    << ", C=" << params.C << "\n"
                    << "Uncertainties: UA=" << params.UA
                    << ", UB=" << params.UB
                    << ", UC=" << params.UC << "\n";

            return params;
    }
        

    static FitParams least_squares_wls_no_D(const std::vector<TestTimeIFs>& test_list,
                                    double phi,
                                    double frac_sigma = 0.03){
            // Build design matrix on log(IDT), same as your code
            const int n = static_cast<int>(test_list.size());
            const int p = 3;

            Eigen::MatrixXd A(n, p);
            Eigen::VectorXd b(n);
            for (int i = 0; i < n; ++i) {
                A(i, 0) = 1.0;
                A(i, 1) = 1.0 / test_list[i].T5;
                A(i, 2) = std::log(test_list[i].P5);
                b(i)    = std::log(test_list[i].IDT / phi);
            }

            
            // --- WLS with constant sigma in log-space ---
            // sigma_log = frac_sigma  ==> w = 1 / sigma^2 = 1 / (frac_sigma^2)
            const double sigma_log = std::max(frac_sigma, 1e-12);
            const double w_scalar  = 1.0 / (sigma_log * sigma_log);
            const double ws        = std::sqrt(w_scalar); // row scale

            // Pre-whiten (constant row scale)
            Eigen::MatrixXd Aw = A * ws;
            Eigen::VectorXd bw = b * ws;

            // Solve (robust SVD, same as your code)
            Eigen::VectorXd x = Aw.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(bw);

            // Residuals (in original log-space)
            Eigen::VectorXd y_pred = A * x;
            Eigen::VectorXd r = b - y_pred;

            // With absolute sigma supplied (like scipy absolute_sigma=True):
            // s^2 = 1, covariance scales purely by (Aᵀ W A)^(-1)
            const double s2 = 1.0;
        
            
            // Covariance via SVD of Aw (AwᵀAw = Aᵀ W A)
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(Aw, Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::VectorXd s = svd.singularValues();
            Eigen::MatrixXd V = svd.matrixV();

            Eigen::VectorXd inv_s2 = s.array().inverse().square();
            for (int i = 0; i < inv_s2.size(); ++i)
                if (!std::isfinite(inv_s2[i]) || s[i] < 1e-12) inv_s2[i] = 0.0;

            // (Aᵀ W A)^+ = V * diag(1/s^2) * Vᵀ
            Eigen::MatrixXd XtWX_pinv = V * inv_s2.asDiagonal() * V.transpose();
            Eigen::MatrixXd cov_beta  = s2 * XtWX_pinv;
            Eigen::VectorXd se        = cov_beta.diagonal().array().sqrt();

            // Back-transform A from log-space and propagate its SE
            x[0]  = std::exp(x[0]);
            double se_logA = se[0];
            double UA      = x[0] * se_logA;

            FitParams params;
            params.A  = x[0];
            params.B  = x[1];
            params.C  = x[2];
            // propagate SE for A with first-order approx σ_A = (dA/dβ0) * σ_β0, with ​​A = exp(β0)
            params.UA = UA;
            params.UB = se[1];
            params.UC = se[2];
        

            std::cout << "Fitted parameters: A=" << params.A
                    << ", B=" << params.B
                    << ", C=" << params.C << "\n"
                    << "Uncertainties: UA=" << params.UA
                    << ", UB=" << params.UB
                    << ", UC=" << params.UC << "\n";

            return params;
    }


    static FitParams least_squares_wls(const std::vector<TestTimeIFs>& test_list,
                                double phi,
                                double frac_sigma = 0.03){
        const int n = static_cast<int>(test_list.size());
        const int p = 4;

        Eigen::MatrixXd A(n, p);
        Eigen::VectorXd b(n);
        for (int i = 0; i < n; ++i) {
            A(i, 0) = 1.0;
            A(i, 1) = 1.0 / test_list[i].T5;
            A(i, 2) = std::log(test_list[i].P5);
            A(i, 3) = std::log(phi);
            b(i)    = std::log(test_list[i].IDT);
        }

        
        // --- WLS with constant sigma in log-space ---
        // sigma_log = frac_sigma  ==> w = 1 / sigma^2 = 1 / (frac_sigma^2)
        const double sigma_log = std::max(frac_sigma, 1e-12);
        const double w_scalar  = 1.0 / (sigma_log * sigma_log);
        const double ws        = std::sqrt(w_scalar); // row scale

        // Pre-whiten (constant row scale)
        Eigen::MatrixXd Aw = A * ws;
        Eigen::VectorXd bw = b * ws;

        // Solve (robust SVD, same as your code)
        Eigen::VectorXd x = Aw.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(bw);

        // Residuals (in original log-space)
        Eigen::VectorXd y_pred = A * x;
        Eigen::VectorXd r = b - y_pred;

        // With absolute sigma supplied (like scipy absolute_sigma=True):
        // s^2 = 1, covariance scales purely by (Aᵀ W A)^(-1)
        const double s2 = 1.0;
    
        
        // Covariance via SVD of Aw (AwᵀAw = Aᵀ W A)
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(Aw, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd s = svd.singularValues();
        Eigen::MatrixXd V = svd.matrixV();

        Eigen::VectorXd inv_s2 = s.array().inverse().square();
        for (int i = 0; i < inv_s2.size(); ++i)
            if (!std::isfinite(inv_s2[i]) || s[i] < 1e-12) inv_s2[i] = 0.0;

        // (Aᵀ W A)^+ = V * diag(1/s^2) * Vᵀ
        Eigen::MatrixXd XtWX_pinv = V * inv_s2.asDiagonal() * V.transpose();
        Eigen::MatrixXd cov_beta  = s2 * XtWX_pinv;
        Eigen::VectorXd se        = cov_beta.diagonal().array().sqrt();

        // Back-transform A from log-space and propagate its SE
        x[0]  = std::exp(x[0]);
        double se_logA = se[0];
        double UA      = x[0] * se_logA;

        FitParams params;
        params.A  = x[0];
        params.B  = x[1];
        params.C  = x[2];
        params.D  = x[3];
        // propagate SE for A with first-order approx σ_A = (dA/dβ0) * σ_β0, with ​​A = exp(β0)
        params.UA = UA;
        params.UB = se[1];
        params.UC = se[2];
        params.UD = se[3];

        std::cout << "Fitted parameters: A=" << params.A
                << ", B=" << params.B
                << ", C=" << params.C
                << ", D=" << params.D << "\n"
                << "Uncertainties: uA=" << params.UA
                << ", uB=" << params.UB
                << ", uC=" << params.UC
                << ", uD=" << params.UD << "\n";

        return params;
    }
    
    static FitParams least_squares(const std::vector<TestTimeIFs>& test_list, double phi) {

        // Phi is always constant for a given experiment, so A and D are linearly dependent, 
        // so no unique solution for both, and uncertainties for A and D dont make sense.
        // so I set A = 1,UA = 0,  and only fit B,C,D 

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(test_list.size(), 4);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(test_list.size());

        for (size_t i = 0; i < test_list.size(); ++i) {

            A(i, 0) = 1.0;
            A(i, 1) = 1.0 / test_list[i].T5; 
            A(i, 2) = std::log(test_list[i].P5);
            A(i, 3) = std::log(phi);
            b(i) = std::log(test_list[i].IDT);
        }

        int n = static_cast<int>(A.rows());
        int p = static_cast<int>(A.cols());

        // Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);

        Eigen::VectorXd x = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

        Eigen::VectorXd y_pred = A * x;
        Eigen::VectorXd residuals = b - y_pred;
        double s2 = (residuals.squaredNorm()) / (n - p); // residual variance

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd s = svd.singularValues();
        Eigen::MatrixXd V = svd.matrixV();

        // Build (AᵀA)⁻¹ = V * diag(1/s²) * Vᵀ using only non-zero singular values
        Eigen::VectorXd inv_s2 = s.array().inverse().square();
        for (int i = 0; i < inv_s2.size(); ++i)
        if (!std::isfinite(inv_s2[i]) || s[i] < 1e-12) inv_s2[i] = 0.0;  // truncate tiny modes

        Eigen::MatrixXd XtX_pinv = V * inv_s2.asDiagonal() * V.transpose();
        Eigen::MatrixXd cov_beta = s2 * XtX_pinv;
        Eigen::VectorXd se = cov_beta.diagonal().array().sqrt();


        x[0] = std::exp(x[0]); // Convert back from log-space for A
     
        FitParams params;
        params.A = x[0];
        params.B = x[1];
        params.C = x[2];
        params.D = x[3];
        // propagate SE for A with first-order approx σ_A = (dA/dβ0) * σ_β0, with ​​A = exp(β0)
        params.UA = x[0] * se[0];
        params.UB = se[1];
        params.UC = se[2];
        params.UD = se[3];

        std::cout << "Fitted parameters: A=" << params.A 
                << ", B=" << params.B
                << ", C=" << params.C
                << ", D=" << params.D << "\n"
                << "Uncertainties: UA=" << params.UA
                << ", UB=" << params.UB
                << ", UC=" << params.UC
                << ", UD=" << params.UD;
        return params;
    }

    static FitParams least_squares_no_D(const std::vector<TestTimeIFs>& test_list, double phi) {

        // Phi is always constant for a given experiment, so A and D are linearly dependent, 
        // so no unique solution for both, and uncertainties for A and D dont make sense.


        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(test_list.size(), 3);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(test_list.size());

        for (size_t i = 0; i < test_list.size(); ++i) {
            A(i, 0) = 1.0;
            A(i, 1) = 1.0 / test_list[i].T5; 
            A(i, 2) = std::log(test_list[i].P5);
            b(i) = std::log(test_list[i].IDT / phi);
        }

        int n = static_cast<int>(A.rows());
        int p = static_cast<int>(A.cols());

        // Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);

        Eigen::VectorXd x = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

        Eigen::VectorXd y_pred = A * x;
        Eigen::VectorXd residuals = b - y_pred;
        double s2 = (residuals.squaredNorm()) / (n - p); // residual variance

        // // --- Covariance matrix of coefficients ---
        // Eigen::MatrixXd XtX_inv = (A.transpose() * A).inverse();
        // Eigen::MatrixXd cov_beta = s2 * XtX_inv;

        // // --- Standard errors ---
        // Eigen::VectorXd se = cov_beta.diagonal().array().sqrt();

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd s = svd.singularValues();
        Eigen::MatrixXd V = svd.matrixV();

        // Build (AᵀA)⁻¹ = V * diag(1/s²) * Vᵀ using only non-zero singular values
        Eigen::VectorXd inv_s2 = s.array().inverse().square();
        for (int i = 0; i < inv_s2.size(); ++i)
        if (!std::isfinite(inv_s2[i]) || s[i] < 1e-12) inv_s2[i] = 0.0;  // truncate tiny modes

        Eigen::MatrixXd XtX_pinv = V * inv_s2.asDiagonal() * V.transpose();
        Eigen::MatrixXd cov_beta = s2 * XtX_pinv;
        Eigen::VectorXd se = cov_beta.diagonal().array().sqrt();
    
     
        x[0] = std::exp(x[0]); // Convert back from log-space for A

        FitParams params;
        params.A = x[0];
        params.B = x[1];
        params.C = x[2];
        // propagate SE for A with first-order approx σ_A = (dA/dβ0) * σ_β0, with ​​A = exp(β0)
        params.UA = x[0] * se[0];  
        params.UB = se[1];
        params.UC = se[2];

        std::cout << "Fitted parameters: A=" << params.A 
                << ", B=" << params.B
                << ", C=" << params.C << "\n"
                << "Uncertainties: UA=" << params.UA
                << ", UB=" << params.UB
                << ", UC=" << params.UC;
        return params;
    }

    static FitParams nonlinear_least_squares_centered_no_D(const std::vector<TestTimeIFs>& test_list, double phi, double T_ref, double P_ref, double relative_measurement_error) {
        FitParams fit_params;
        
        std::vector<double> T5(test_list.size());
        std::vector<double> P5(test_list.size());
        std::vector<double> tau(test_list.size());

        for (size_t i = 0; i < test_list.size(); ++i) {
            T5[i] = test_list[i].T5;
            P5[i] = test_list[i].P5;
            tau[i] = test_list[i].IDT;
        }

        size_t N = T5.size();
        // Reference values: median
        //TestTimeIFs medians = median(test_list);
        double Tref = T_ref;
        double Pref = P_ref;

        std::cout << "Reference conditions: Tref=" << Tref << ", Pref=" << Pref << std::endl;

        double A0 = 1;
        double B0 = 1;
        double C0 = 1;
        double params[3] = {A0, B0, C0};

        ceres::Problem problem;
        for (size_t i = 0; i < N; ++i) {
            //weight by 1/variance: sigma2 = (relative_measurement_error * tau_i)^2, std assumed proportional to IDT
            double sigma2 = std::pow(relative_measurement_error * tau[i], 2);

            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<IgnitionResidual, 1, 3>(
                    new IgnitionResidual(T5[i], P5[i], phi, tau[i], Tref, Pref, sigma2)
                ),
                nullptr,
                params
            );
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 20000;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << summary.FullReport() << std::endl;
        std::cout << "Fitted parameters:" << std::endl;
        std::cout << "A = " << params[0] << std::endl;
        std::cout << "B = " << params[1] << std::endl;
        std::cout << "C = " << params[2] << std::endl;

        ceres::Covariance::Options cov_options;
        ceres::Covariance covariance(cov_options);

        std::vector<const double*> parameter_blocks = { params };
        if (!covariance.Compute(parameter_blocks, &problem)) {
            std::cerr << "Covariance computation failed!" << std::endl;
        } else {
            double cov[3*3];
            covariance.GetCovarianceBlock(params, params, cov);

            fit_params.UA = 2 * std::sqrt(cov[0]);
            fit_params.UB = 2 * std::sqrt(cov[4]);  
            fit_params.UC = 2 * std::sqrt(cov[8]);

            std::cout << "Parameter uncertainties (2-sigma):" << std::endl;
            std::cout << "A sigma = " << 2 * std::sqrt(cov[0]) << std::endl;
            std::cout << "B sigma = " << 2 * std::sqrt(cov[4]) << std::endl;
            std::cout << "C sigma = " << 2 * std::sqrt(cov[8]) << std::endl;
        }

        
        fit_params.A = params[0];
        fit_params.B = params[1];
        fit_params.C = params[2];
        fit_params.D = 0.0;
        fit_params.UD = 0.0;

        double chi2 = 0.0;
        for (size_t i = 0; i < T5.size(); ++i) {
            double model = params[0] * std::exp(params[1] * (1.0/T5[i] - 1.0/Tref))
                        * std::pow(P5[i]/Pref, params[2])
                        * phi;

            double weight = std::pow(relative_measurement_error * tau[i], 2);
            double resid = (model - tau[i]) / weight;
            chi2 += resid * resid;
        }

        int dof = static_cast<int>(T5.size()) - 3; // 3 fitted params: A,B,C
        double red_chi2 = chi2 / dof;

        std::cout << "Chi² = " << chi2 << std::endl;
        std::cout << "Reduced Chi² = " << red_chi2 << std::endl;

        return fit_params;
    }
        
    static void function_cx_1_func_no_D(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr) 
    {
        // this callback calculates f(c,x)
        // where x is a position on X-axis and c is adjustable parameter
        func = c[0] * exp(c[1]/x[0]) * pow(x[1],c[2]) * x[2];
    }
    static void function_cx_1_grad_no_D(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr) 
    {
        // this callback calculates f(c,x) and gradient G={df/dc[i]}
        // where x is a position on X-axis and c is adjustable parameter.
        // IMPORTANT: gradient is calculated with respect to C, not to X
        func = c[0] * exp(c[1]/x[0]) * pow(x[1],c[2]) * x[2];
        grad[0] = exp(c[1]/x[0]) * pow(x[1],c[2]) * x[2];
        grad[1] = c[0] * exp(c[1]/x[0]) * pow(x[1],c[2]) * x[2] / x[0];
        grad[2] = c[0] * exp(c[1]/x[0]) * pow(x[1],c[2]) * x[2] * log(x[1]);
    }

    static FitParams least_squares_no_D_alglib(const std::vector<TestTimeIFs>& test_list, double phi) {
        
        using namespace alglib;

        const ae_int_t n = static_cast<ae_int_t>(test_list.size());
        const ae_int_t p = 3; // [intercept, x, z]

        // Build design matrix F (n x p): column0=1 (intercept), column1=x, column2=z
        real_2d_array F;
        F.setlength(n, p);
        for (ae_int_t i = 0; i < n; ++i) {
            F[i][0] = 1.0;
            F[i][1] = 1.0 / test_list[i].T5;
            F[i][2] = std::log(test_list[i].P5);
        }

        real_1d_array y;
        y.setlength(n);
        for (ae_int_t i = 0; i < n; ++i) y[i] = std::log(test_list[i].IDT / phi);


        // Fit
        real_1d_array c;       // coefficients (size p)
        lsfitreport rep;       // diagnostics

        lsfitlinear(y, F, n, p, c, rep);  // OLS

        if (rep.terminationtype != 1) {
            std::cerr << "LS fit failed, info=" << rep.terminationtype << "\n";
        }

        std::cout << "Coefficients:\n";
        std::cout << "  intercept = " << c[0] << "\n";
        std::cout << "  x_1         = " << c[1] << "\n";
        std::cout << "  x_2         = " << c[2] << "\n\n";

        std::cout << "Diagnostics:\n";
        std::cout << "  RMSE      = " << rep.rmserror << "\n";
        std::cout << "  R^2       = " << rep.r2 << "\n";         // available in recent ALGLIB
        std::cout << "  cond^-1   = " << rep.taskrcond << "\n\n";  // reciprocal condition number

        c[0] = std::exp(c[0]); // Convert back from log-space for A

        FitParams params;
        params.A = c[0];
        params.B = c[1];
        params.C = c[2];
        // propagate SE for A with first-order approx σ_A = (dA/dβ0) * σ_β0, with ​​A = exp(β0)
        params.UA = c[0] * rep.errpar[0];  
        params.UB = rep.errpar[1];
        params.UC = rep.errpar[2];

        std::cout << "Fitted parameters: A=" << params.A 
                << ", B=" << params.B
                << ", C=" << params.C << "\n"
                << "Uncertainties: UA=" << params.UA
                << ", UB=" << params.UB
                << ", UC=" << params.UC;
        return params;
    }

    static FitParams least_squares_wls_no_D_alglib(const std::vector<TestTimeIFs>& test_list,
                                    double phi,
                                    double frac_sigma = 0.032){
                                        //!!!!!!!!!!!!!!!!!!!Doesnt Work!!!!!!!!!!!!!!!!!!!
        using namespace alglib;

        const ae_int_t n = static_cast<ae_int_t>(test_list.size());
        const ae_int_t p = 3; // [intercept, x, z]

        // Build design matrix F (n x p): column0=1 (intercept), column1=x, column2=z
        real_2d_array F;
        F.setlength(n, p);
        for (ae_int_t i = 0; i < n; ++i) {
            F[i][0] = 1.0;
            F[i][1] = 1.0 / test_list[i].T5;
            F[i][2] = std::log(test_list[i].P5);
        }

        real_1d_array y;
        y.setlength(n);
        for (ae_int_t i = 0; i < n; ++i) y[i] = std::log(test_list[i].IDT / phi);

        // const double sigma_log = std::max(frac_sigma, 1e-12);
        // const double w_scalar  = 1.0 / (sigma_log * sigma_log);
        // const double ws        = std::sqrt(w_scalar); // row scale
        real_1d_array w;
        w.setlength(n);
        for (ae_int_t i = 0; i < n; ++i) w[i] = frac_sigma * y[i];

        // Fit
        real_1d_array c;       // coefficients (size p)
        lsfitreport rep;       // diagnostics

        lsfitlinearw(y, w, F, n, p, c, rep);  // OLS

        if (rep.terminationtype != 1) {
            std::cerr << "LS fit failed, info=" << rep.terminationtype << "\n";
        }

        std::cout << "Coefficients:\n";
        std::cout << "  intercept = " << c[0] << "\n";
        std::cout << "  x_1         = " << c[1] << "\n";
        std::cout << "  x_2         = " << c[2] << "\n\n";

        std::cout << "Diagnostics:\n";
        std::cout << "  RMSE      = " << rep.rmserror << "\n";
        std::cout << "  R^2       = " << rep.r2 << "\n";         // available in recent ALGLIB
        std::cout << "  cond^-1   = " << rep.taskrcond << "\n\n";  // reciprocal condition number

        c[0] = std::exp(c[0]); // Convert back from log-space for A

        FitParams params;
        params.A = c[0];
        params.B = c[1];
        params.C = c[2];
        // propagate SE for A with first-order approx σ_A = (dA/dβ0) * σ_β0, with ​​A = exp(β0)
        params.UA = c[0] * rep.errpar[0];  
        params.UB = rep.errpar[1];
        params.UC = rep.errpar[2];

        std::cout << "Fitted parameters: A=" << params.A 
                << ", B=" << params.B
                << ", C=" << params.C << "\n"
                << "Uncertainties: UA=" << params.UA
                << ", UB=" << params.UB
                << ", UC=" << params.UC << "\n";
        return params;
    }

    static FitParams nonlinear_least_squares_no_D_alglib(const std::vector<TestTimeIFs>& test_list, double phi) {
        FitParams params;
        try{
            const ae_int_t n = static_cast<ae_int_t>(test_list.size());
            const ae_int_t p = 3; // [intercept, x, z]

            // Build design matrix F (n x p): column0=1 (intercept), column1=x, column2=z
            real_2d_array x;
            x.setlength(n, p);
            for (ae_int_t i = 0; i < n; ++i) {
                x[i][0] = 1.0;
                x[i][1] = 1.0 / test_list[i].T5;
                x[i][2] = std::log(test_list[i].P5);
            }

            real_1d_array y;
            y.setlength(n);
            for (ae_int_t i = 0; i < n; ++i) y[i] = std::log(test_list[i].IDT / phi);

            // Fit
            real_1d_array c;       // coefficients (size p)
            lsfitreport rep;       // diagnostics
            lsfitstate state;
            ae_int_t maxits = 0;
            double epsx = 0.000001;

            //
            // Fitting without weights
            //
            lsfitcreatefg(x, y, c, state);
            lsfitsetcond(state, epsx, maxits);
            lsfitfit(state, function_cx_1_func_no_D, function_cx_1_grad_no_D);
            lsfitresults(state, c, rep);
            printf("%d\n", int(rep.terminationtype)); // EXPECTED: 2
            printf("%s\n", c.tostring(1).c_str()); // EXPECTED: [1.5]



            std::cout << "Coefficients:\n";
            std::cout << "  intercept = " << c[0] << "\n";
            std::cout << "  x_1         = " << c[1] << "\n";
            std::cout << "  x_2         = " << c[2] << "\n\n";

            std::cout << "Diagnostics:\n";
            std::cout << "  RMSE      = " << rep.rmserror << "\n";
            std::cout << "  R^2       = " << rep.r2 << "\n";         // available in recent ALGLIB
            std::cout << "  cond^-1   = " << rep.taskrcond << "\n\n";  // reciprocal condition number

            c[0] = std::exp(c[0]); // Convert back from log-space for A

            params.A = c[0];
            params.B = c[1];
            params.C = c[2];
            // propagate SE for A with first-order approx σ_A = (dA/dβ0) * σ_β0, with ​​A = exp(β0)
            params.UA = c[0] * rep.errpar[0];  
            params.UB = rep.errpar[1];
            params.UC = rep.errpar[2];

            std::cout << "Fitted parameters: A=" << params.A 
                    << ", B=" << params.B
                    << ", C=" << params.C << "\n"
                    << "Uncertainties: UA=" << params.UA
                    << ", UB=" << params.UB
                    << ", UC=" << params.UC;
        } catch(alglib::ap_error alglib_exception){
             printf("ALGLIB exception with message '%s'\n", alglib_exception.msg.c_str());
        }
        return params;
    }


    static TestTimeIFs median(std::vector<TestTimeIFs> data) {
        if (data.empty()) return {0.0, 0.0, 0.0};

        auto extract_and_median = [](auto get, std::vector<TestTimeIFs>& v) {
            std::vector<double> vals;
            vals.reserve(v.size());
            for (auto& x : v) vals.push_back(get(x));

            size_t n = vals.size();
            std::nth_element(vals.begin(), vals.begin() + n / 2, vals.end());
            double med = vals[n / 2];
            if (n % 2 == 0) {
                std::nth_element(vals.begin(), vals.begin() + n / 2 - 1, vals.end());
                med = (med + vals[n / 2 - 1]) / 2.0;
            }
            return med;
        };

        TestTimeIFs medians;
        medians.IDT = extract_and_median([](const TestTimeIFs& t){ return t.IDT; }, data);
        medians.T5  = extract_and_median([](const TestTimeIFs& t){ return t.T5;  }, data);
        medians.P5  = extract_and_median([](const TestTimeIFs& t){ return t.P5;  }, data);
        return medians;
    }
    static double calc_u_tau(double T5, double P5, double phi, double uT5, double uP5, double uphi, const FitParams& params) {
        double x1 = T5;
        double x2 = P5;
        double x3 = phi;

        double u1 = uT5;
        double u2 = uP5;
        double u3 = uphi;

        double A = params.A;
        double B = params.B;
        double C = params.C;
        double D = params.D;

        double U_A = params.UA;
        double U_B = params.UB;
        double U_C = params.UC;
        double U_D = params.UD;

        double U_x1 = x1 * u1;
        double U_x2 = x2 * u2;
        double U_x3 = x3 * u3;

        double usqr_x1 = (U_x1/2) * (U_x1/2);
        double usqr_x2 = (U_x2/2) * (U_x2/2);
        double usqr_x3 = (U_x3/2) * (U_x3/2);
        double usqr_xA = (U_A/2) * (U_A/2);
        double usqr_xB = (U_B/2) * (U_B/2);
        double usqr_xC = (U_C/2) * (U_C/2);
        double usqr_xD = (U_D/2) * (U_D/2);

        double exp_term  = std::exp(B / x1);
        double base_term = exp_term * std::pow(x2, C) * std::pow(x3, D);

        double q1 = std::pow(A * base_term * (-B / (x1 * x1)), 2) * usqr_x1;
        double q2 = std::pow(C * A * exp_term * std::pow(x2, C - 1) * std::pow(x3, D), 2) * usqr_x2;
        double q3 = std::pow(D * A * exp_term * std::pow(x2, C) * std::pow(x3, D - 1), 2) * usqr_x3;
        double q4 = std::pow(exp_term * std::pow(x2, C) * std::pow(x3, D), 2) * usqr_xA;
        double q5 = std::pow((1.0 / x1) * A * exp_term * std::pow(x2, C) * std::pow(x3, D), 2) * usqr_xB;
        double q6 = std::pow(A * exp_term * std::pow(x2, C) * std::pow(x3, D) * std::log(x2), 2) * usqr_xC;
        double q7 = std::pow(A * exp_term * std::pow(x2, C) * std::pow(x3, D) * std::log(x3), 2) * usqr_xD;

        return q1 + q2 + q3 + q4 + q5 + q6 + q7;
    }

    static double calc_u_tau_better(double T5, double P5, double phi, double uT5, double uP5, double uphi, const FitParams& params) {
        double x1 = T5;
        double x2 = P5;
        double x3 = phi;

        double u1 = uT5;
        double u2 = uP5;
        double u3 = uphi;

        double A = params.A;
        double B = params.B;
        double C = params.C;
        double D = params.D;

        double u_A = params.UA;
        double u_B = params.UB;
        double u_C = params.UC;
        double u_D = params.UD;

        //95% uncertainty
        double U_x1 = x1 * u1; 
        double U_x2 = x2 * u2;
        double U_x3 = x3 * u3;

        double usqr_x1 = (U_x1/2) * (U_x1/2);
        double usqr_x2 = (U_x2/2) * (U_x2/2);
        double usqr_x3 = (U_x3/2) * (U_x3/2);
        double usqr_xA = (u_A) * (u_A);
        double usqr_xB = (u_B) * (u_B);
        double usqr_xC = (u_C) * (u_C);
        double usqr_xD = (u_D) * (u_D);

        double exp_term  = std::exp(B / x1);
        double base_term = exp_term * std::pow(x2, C) * std::pow(x3, D);

        double q1 = std::pow(A * base_term * (-B / (x1 * x1)), 2) * usqr_x1;
        double q2 = std::pow(C * A * exp_term * std::pow(x2, C - 1) * std::pow(x3, D), 2) * usqr_x2;
        double q3 = std::pow(D * A * exp_term * std::pow(x2, C) * std::pow(x3, D - 1), 2) * usqr_x3;
        double q4 = std::pow(exp_term * std::pow(x2, C) * std::pow(x3, D), 2) * usqr_xA;
        double q5 = std::pow((1.0 / x1) * A * exp_term * std::pow(x2, C) * std::pow(x3, D), 2) * usqr_xB;
        double q6 = std::pow(A * exp_term * std::pow(x2, C) * std::pow(x3, D) * std::log(x2), 2) * usqr_xC;
        double q7 = std::pow(A * exp_term * std::pow(x2, C) * std::pow(x3, D) * std::log(x3), 2) * usqr_xD;

        return q1 + q2 + q3 + q4 + q5 + q6 + q7;
    }



    static double calc_u_tau_no_D (double T5, double P5, double phi, double uT5, double uP5, double uphi, const FitParams& params) {
        double x1 = T5;
        double x2 = P5;
        double x3 = phi;

        double u1 = uT5;
        double u2 = uP5;
        double u3 = uphi;

        double A = params.A;
        double B = params.B;
        double C = params.C;

        double U_A = params.UA;
        double U_B = params.UB;
        double U_C = params.UC;

        double U_x1 = x1 * u1;
        double U_x2 = x2 * u2;
        double U_x3 = x3 * u3;

        double usqr_x1 = (U_x1/2) * (U_x1/2);
        double usqr_x2 = (U_x2/2) * (U_x2/2);
        double usqr_x3 = (U_x3/2) * (U_x3/2);
        double usqr_xA = (U_A/2) * (U_A/2);
        double usqr_xB = (U_B/2) * (U_B/2);
        double usqr_xC = (U_C/2) * (U_C/2);

        double exp_term  = std::exp(B / x1);
        double base_term = exp_term * std::pow(x2, C) * x3;

        double q1 = std::pow(A * base_term * (-B / (x1 * x1)), 2) * usqr_x1;
        double q2 = std::pow(C * A * exp_term * std::pow(x2, C - 1) * x3, 2) * usqr_x2;
        double q3 = std::pow(A * exp_term * std::pow(x2, C) * x3, 2) * usqr_x3;
        double q4 = std::pow(exp_term * std::pow(x2, C) * x3, 2) * usqr_xA;
        double q5 = std::pow((1.0 / x1) * A * exp_term * std::pow(x2, C) * x3, 2) * usqr_xB;
        double q6 = std::pow(A * exp_term * std::pow(x2, C) * x3 * std::log(x2), 2) * usqr_xC;
        

        return q1 + q2 + q3 + q4 + q5 + q6;
    }

    static double calc_u_tau_no_D_centered (double T5, double P5, double phi, double uT5, double uP5, double uphi, const FitParams& params, double median_T5, double median_P5) {
        double x1 = T5;
        double x2 = P5;
        double x3 = phi;

        double u1 = uT5;
        double u2 = uP5;
        double u3 = uphi;

        double A = params.A;
        double B = params.B;
        double C = params.C;

        double U_A = params.UA;
        double U_B = params.UB;
        double U_C = params.UC;

        double U_x1 = x1 * u1;
        double U_x2 = x2 * u2;
        double U_x3 = x3 * u3;

        double usqr_x1 = (U_x1/2) * (U_x1/2);
        double usqr_x2 = (U_x2/2) * (U_x2/2);
        double usqr_x3 = (U_x3/2) * (U_x3/2);
        double usqr_xA = (U_A/2) * (U_A/2);
        double usqr_xB = (U_B/2) * (U_B/2);
        double usqr_xC = (U_C/2) * (U_C/2);

        double tau = A * std::exp(B *((1/T5) - (1/median_T5))) * std::pow((P5/median_P5), C) * phi;
        
        double q1 = std::pow((-1) * tau * B / (T5 * T5), 2) * usqr_x1;
        double q2 = std::pow(tau * C / P5, 2) * usqr_x2;
        double q3 = std::pow(tau / phi, 2) * usqr_x3;
        double q4 = std::pow(tau/A, 2) * usqr_xA;
        double q5 = std::pow(tau*((1/T5) - (1/median_T5)), 2) * usqr_xB;
        double q6 = std::pow(tau * std::log(P5/median_P5), 2) * usqr_xC;
        

        return q1 + q2 + q3 + q4 + q5 + q6;
    }


    static std::vector<double> calc_random_uncertainties(const std::vector<TestTimeIFs>& test_list, double phi, double uT5, double uP5, double uphi) {
        std::vector<double> random_uncertainties;
        random_uncertainties.reserve(test_list.size());

         //median of T5 and P5 from data
        TestTimeIFs medians = median(test_list);

        //std::cout << "Median T5: " << medians.T5 << ", Median P5: " << medians.P5 << "\n";

        //const FitParams fit_params = least_squares(test_list, phi);
        //const FitParams fit_params {0.0075, 16400.0, 1.91, -3.4, 0.000046, 86.0, 0.071, 0.052 };
        //const FitParams fit_params {6.05891e-05, 16402.4, 1.90166, 0, 7.5e-06, 1.4e+02, 0.13, 0}; nadia nonlinear
        //const FitParams fit_params {0.606167, 7846.46, -0.272313, 0, 1.1, 5.4e+02, 0.58, 0};//davidson nonliner
        //const FitParams fit_params {0.01641, 7838.82954, 0.78694, 1.0, 0.0507, 1264.58038, 1.0564, 0.0};//davidson nonlinear originpro
        //const FitParams fit_params{ 0.0075, 16400.0, 1.91, -3.4, 0.000046, 86.0, 0.071, 0.052}; //nadia params


        //const FitParams fit_params{ 104.46, 16489.6, 1.4507, 0, 9.39666743e+00, 1.08194604e+03, 8.26777475e-01, 0}; //nadia nonlinear centered
        //const FitParams fit_params{ 394.538, 14923.5, -2.07231, 0, 18.0, 8.5e+02, 0.62, 0}; //shen nonlinear centered improved
        //const FitParams fit_params{ 825.41, 6786.81, 0.149807, 0, 1e+02, 3.6e+03, 3.6e-02, 0}; //davidson nonlinear centered final
        const FitParams fit_params = nonlinear_least_squares_centered_no_D(test_list, phi, medians.T5, medians.P5, 0.02);

        //const FitParams fit_params = least_squares_no_D(test_list, phi);
        //const FitParams fit_params = least_squares_wls(test_list, phi, 0.032);

       
    
        for (const auto& entry : test_list) {
            //double u_tau = calc_u_tau_no_D(entry.T5, entry.P5, phi, uT5, uP5, uphi, fit_params);
            double u_tau = calc_u_tau_no_D_centered(entry.T5, entry.P5, phi, uT5, uP5, uphi, fit_params, medians.T5, medians.P5);
            random_uncertainties.push_back(u_tau);
        }
        return random_uncertainties;
    }

    static std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
    calc_final_extended_uncertainties(const std::vector<TestTimeIFs>& test_list,
                                      const std::vector<double>& systematic,
                                      const std::vector<double>& random)
    {
        std::vector<double> combined(systematic.size());
        std::vector<double> final_extended(systematic.size());
        std::vector<double> percent(systematic.size());

        for (size_t i = 0; i < systematic.size(); ++i) {
            combined[i] = systematic[i] + random[i];
            final_extended[i] = std::sqrt(combined[i]) * 2; // k=2 for 95% confidence interval
            percent[i] = final_extended[i] / test_list[i].IDT * 100.0;
        }
        return {combined, final_extended, percent};
    }
};

int main(int argc, char* argv[]) { 

    std::string input_path;
    std::string output_path = "UQ_results.csv"; // default output

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        }
    }
     if (input_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " --input <input_json> [--output <output_csv>]\n";
        return 1;
    }
    if (!std::filesystem::exists(input_path)) {
        std::cerr << "Error: Input file does not exist: " << input_path << "\n";
        return 1;
    }

    UncertaintyQuantification uq(input_path);

    if (!uq.writeCSV(output_path)) {
        std::cerr << "CSV write failed.\n";
        return 1;
    }
    std::cout << "Wrote results to " << output_path << "\n";

}





