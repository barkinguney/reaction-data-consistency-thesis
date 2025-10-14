#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <fstream>
#include <iomanip>
#include <locale>
#include <string>
#include <nlohmann/json.hpp> 
#include <Eigen/Dense>

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

        if (tests.size() != res.percent_uncertainty.size()) return false;

        std::ofstream out(filepath);
        if (!out) return false;

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
        experiment.fixed_IFs.driver_length_m = j["FixedIFs"]["driver_length_m"];
        experiment.fixed_IFs.driven_length_m = j["FixedIFs"]["driven_length_m"];
        experiment.fixed_IFs.driven_inner_diameter_cm = j["FixedIFs"]["driven_inner_diameter_cm"];
        experiment.fixed_IFs.tailoring = j["FixedIFs"]["tailoring"];
        experiment.fixed_IFs.inserts = j["FixedIFs"]["inserts"];
        experiment.fixed_IFs.crv = j["FixedIFs"]["crv"];
        experiment.fixed_IFs.dilution = j["FixedIFs"]["dilution"];
        experiment.fixed_IFs.impurities = j["FixedIFs"]["impurities"];
        experiment.fixed_IFs.measurement_location_cm = j["FixedIFs"]["measurement_location_cm"];
        experiment.fixed_IFs.fuel_air_ratio = j["FixedIFs"]["fuel_air_ratio"];

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

        experiment.phi = j["Experiment"]["phi"];
        experiment.mixture = j["Experiment"]["mixture"];

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
        params.D = x[3];
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

    static FitParams least_squares2(const std::vector<TestTimeIFs>& test_list, double phi) {

        // Phi is always constant for a given experiment, so A and D are linearly dependent, 
        // so no unique solution for both, and uncertainties for A and D dont make sense.
        // so I set A = 1,UA = 0,  and only fit B,C,D 

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(test_list.size(), 3);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(test_list.size());

        for (size_t i = 0; i < test_list.size(); ++i) {
            A(i, 0) = 1.0 / test_list[i].T5; 
            A(i, 1) = std::log(test_list[i].P5);
            A(i, 2) = std::log(phi);
            b(i) = std::log(test_list[i].IDT);
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
    
     
        FitParams params;
        params.A = 1;
        params.B = x[0];
        params.C = x[1];
        params.D = x[2];
        params.UA = 0;
        params.UB = se[0];
        params.UC = se[1];
        params.UD = se[2];

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

    static std::vector<double> calc_random_uncertainties(const std::vector<TestTimeIFs>& test_list, double phi, double uT5, double uP5, double uphi) {
        std::vector<double> random_uncertainties;
        random_uncertainties.reserve(test_list.size());
        const FitParams fit_params = least_squares2(test_list, phi); 

        for (const auto& entry : test_list) {
            double u_tau = calc_u_tau(entry.T5, entry.P5, phi, uT5, uP5, uphi, fit_params);
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

int main() {
    std::string filename = "davidson_2005";
    UncertaintyQuantification uq(std::string(PROJECT_ROOT) + "/data/" + filename + ".json");
    
    if (!uq.writeCSV(std::string(PROJECT_ROOT) + "/data/"+ filename + "_UQ_results.csv")) {
        std::cerr << "CSV write failed.\n";
        return 1;
    }
    std::cout << "Wrote " << filename << "_UQ_results.csv\n";
}





