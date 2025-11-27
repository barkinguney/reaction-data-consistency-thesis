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
#include <ceres/ceres.h>


using json = nlohmann::json;

class UncertaintyQuantification {
public:
    // Fixed impacting factors (IFs) for an experiment
    struct FixedIFs {
        double driver_length_m;             
        double driven_length_m;            
        double driven_inner_diameter_cm;   
        double measurement_location_cm;     
        bool tailoring;                     
        bool inserts;                       
        bool crv;                          
        bool dilution;                      
        bool impurities;                    
        bool fuel_air_ratio; // true if phi > 0.3
    };

    struct TestTimeIFs {
        double IDT;         // Ignition delay time[μs]
        double T5;          // Temperature [K]
        double P5;          // Pressure [atm]
    };

    struct FitParams {
        double A, B, C, D;     // regression parameters
        double UA, UB, UC, UD; // uncertainties of regression parameters
    };

    struct Experiment {
        FixedIFs fixed_IFs;    // Fixed IFs for the experiment
        std::vector<TestTimeIFs> test_time_IF_list;  // Test time IFs for the experiment
        double phi;            // Stoichiometric ratio
        double uT5, uP5, uphi;  // fractional uncertainties
    };

    struct Results {
        std::vector<double> systematic_uncertainty;
        std::vector<double> random_uncertainty;
        std::vector<double> combined_uncertainty;
        std::vector<double> final_expanded_uncertainty;
        std::vector<double> percent_uncertainty;
    };

    
    struct IDTResidual {
        /*
        * Model and the residual for Ceres Solver
        * Ceres does automatic differentiation
        * if your library does not support automatic differentiation, you need to provide the analytic Jacobian as well

        * Model: tau = A * exp(B * (1/T - 1/Tref)) * (P/Pref)^C * phi
        *
        * Jacobian w.r.t parameters [A, B, C]:
        *
        * d(tau)/dA = exp(B * (1/T - 1/Tref)) * (P/Pref)^C * phi
        *
        * d(tau)/dB = A * (1/T - 1/Tref) * exp(B * (1/T - 1/Tref)) * (P/Pref)^C * phi
        *
        * d(tau)/dC = A * exp(B * (1/T - 1/Tref)) * (P/Pref)^C * ln(P/Pref) * phi
        */
        IDTResidual(double T, double P, double phi, double tau, double Tref, double Pref, double weight = 1.0)
            : T(T), P(P), phi(phi), tau(tau), Tref(Tref), Pref(Pref), weight(weight) {}

        template <typename Ttype>
        bool operator()(const Ttype* const params, Ttype* residual) const {
            Ttype Aref = params[0];
            Ttype B    = params[1];
            Ttype C    = params[2];

            Ttype model = Aref * ceres::exp(B * (Ttype(1.0)/T - Ttype(1.0)/Tref)) 
                        * ceres::pow(Ttype(P)/Pref, C) 
                        * Ttype(phi);

            residual[0] = ceres::sqrt(Ttype(weight)) * (model - Ttype(tau)); // weighted residual
            return true;
        }

        double T, P, phi, tau, Tref, Pref, weight;
    };

    // Constructor reads JSON input file
    UncertaintyQuantification(const std::string& filepath) 
        : experiment(readJSON(filepath)) {}

    // Main function to compute all uncertainties
    static Results do_uncertainty_quantification(const Experiment& exp) {
        Results out;

        auto [systematic_uncertainty, systematic_relative_errors] =
            calc_systematic_uncertainties(mu_ideal_fixed_IFs_bounds_, 
                                          exp.fixed_IFs,
                                          exp.test_time_IF_list,
                                          mu_ideal_bounds_test_time_IF_);

        out.systematic_uncertainty = std::move(systematic_uncertainty);

        out.random_uncertainty =
            calc_random_uncertainties(exp.test_time_IF_list, exp.phi, exp.uT5, exp.uP5, exp.uphi, systematic_relative_errors);

        auto [combined, final_ext, percent] =
            calc_final_expanded_uncertainties(exp.test_time_IF_list,
                                              out.systematic_uncertainty,
                                              out.random_uncertainty);
        
        out.combined_uncertainty = std::move(combined);
        out.final_expanded_uncertainty = std::move(final_ext);
        out.percent_uncertainty = std::move(percent);

        return out;
    }

    // Write results to CSV
    bool writeCSV(const std::string& filepath) const {
        auto res = do_uncertainty_quantification(experiment);
        const auto& tests = experiment.test_time_IF_list;  

        if (tests.size() != res.percent_uncertainty.size()) {
            std::cerr << "Mismatch in test size and results size" << tests.size() << " vs " << res.percent_uncertainty.size() << "\n";   
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
            "systematic,random,combined,final_expanded,expanded_relative\n";

        for (size_t i = 0; i < tests.size(); ++i) {
            out << tests[i].IDT << ','
                << tests[i].T5        << ','
                << tests[i].P5        << ','
                << res.systematic_uncertainty[i]      << ','
                << res.random_uncertainty[i]          << ','
                << res.combined_uncertainty[i]        << ','
                << res.final_expanded_uncertainty[i]  << ','
                << res.percent_uncertainty[i]
                << '\n';
        }
        return true;
    }

   
private:
    inline static double delta_0_ = 5.0;

    // Ideal bounds (min/max) for fixed IFs and test-time IFs
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
        try {
            file >> j;
        } catch (const json::parse_error& e) {
            throw std::runtime_error("JSON parse error: " + std::string(e.what()));
        }
    
        Experiment experiment;
        try {
            //assumes phi is the same for all samples 
            experiment.phi = j.at("Experiment").at("phi").get<double>();

            auto& fixed = j.at("FixedIFs");
            experiment.fixed_IFs.driver_length_m = fixed.at("driver_length_m").get<double>();
            experiment.fixed_IFs.driven_length_m = fixed.at("driven_length_m").get<double>();
            experiment.fixed_IFs.driven_inner_diameter_cm = fixed.at("driven_inner_diameter_m").get<double>() * 100.0;
            experiment.fixed_IFs.tailoring = fixed.at("tailoring").get<bool>();
            experiment.fixed_IFs.inserts = fixed.at("inserts").get<bool>();
            experiment.fixed_IFs.crv = fixed.at("crv").get<bool>();
            experiment.fixed_IFs.dilution = fixed.at("dilution").get<bool>();
            experiment.fixed_IFs.impurities = fixed.at("impurities").get<bool>();
            experiment.fixed_IFs.measurement_location_cm = fixed.at("measurement_location_m").get<double>() * 100.0;
            experiment.fixed_IFs.fuel_air_ratio = experiment.phi > 0.3;

            experiment.uT5 = j.at("Experiment").at("T5_uncertainty").get<double>();
            experiment.uP5 = j.at("Experiment").at("P5_uncertainty").get<double>();
            experiment.uphi = j.at("Experiment").at("phi_uncertainty").get<double>();

            for (const auto& entry : j.at("Experiment").at("TestTimeIFsList")) {
                TestTimeIFs test_time_IF;
                test_time_IF.IDT = entry.at("IDT").get<double>();
                test_time_IF.T5 = entry.at("T5").get<double>();
                test_time_IF.P5 = entry.at("P5").get<double>();
                experiment.test_time_IF_list.push_back(test_time_IF);
            }

        } catch (const json::out_of_range& e) {
            throw std::runtime_error("Missing key in JSON: " + std::string(e.what()));
        } catch (const json::type_error& e) {
            throw std::runtime_error("Wrong type in JSON: " + std::string(e.what()));
        }

        return experiment;
    }
    // Harrington desirability function to compute systematic error contribution
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
    // Harrington desirability function for boolean IFs
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

    static std::tuple<std::vector<double>, std::vector<double>> calc_systematic_uncertainties(const std::vector<FixedIFs>& fixed_IF_bounds, 
                                                      const FixedIFs& actual_fixed,
                                                      const std::vector<TestTimeIFs>& test_list,
                                                      const std::vector<TestTimeIFs>& ideal_bounds){
        double fixed_IF_error = calc_total_fixed_harrington(fixed_IF_bounds, actual_fixed);

        std::vector<double> systematic_uncertainties;
        systematic_uncertainties.reserve(test_list.size());

        std::vector<double> systematic_relative_errors;
        systematic_relative_errors.reserve(test_list.size());

        //Only use FixedIFs and IDT for systematic uncertainty. T5,P5 not necessary 
        double lower_bound = ideal_bounds.front().IDT;
        double upper_bound = ideal_bounds.back().IDT;

        for (const auto& entry : test_list) {
            double mu_actual = entry.IDT;
            double delta_val = 0.0;

            delta_val = harrington_desirability(lower_bound, upper_bound, mu_actual);
            double systematic_error = delta_val + fixed_IF_error + delta_0_;
            systematic_relative_errors.push_back(systematic_error);
            double systematic_uncertainty = std::pow((entry.IDT * systematic_error / 100.0 / 2.0), 2);
            systematic_uncertainties.push_back(systematic_uncertainty);
        }

        return {systematic_uncertainties, systematic_relative_errors};
    }

    // Fit IDT model using nonlinear least squares
    static FitParams nonlinear_least_squares_centered_no_D(const std::vector<TestTimeIFs>& test_list, double phi, double T_ref, double P_ref, const std::vector<double>& systematic_relative_errors) {
        /*
        * model = Aref * exp(B*(1/T - 1/Tref)) * (P/Pref)^C * phi
        */
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
        // Reference values, median
        double Tref = T_ref;
        double Pref = P_ref;


        double A0 = 1;
        double B0 = 1;
        double C0 = 1;
        double params[3] = {A0, B0, C0};
        
        ceres::Problem problem;
        for (size_t i = 0; i < N; ++i) {
            
            //wls weight by 1/variance
            double sigma = systematic_relative_errors[i] * tau[i] / 100.0 / 2.0;
            double weight = 1.0 / std::pow(sigma, 2);

            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<IDTResidual, 1, 3>(
                    new IDTResidual(T5[i], P5[i], phi, tau[i], Tref, Pref, weight)
                ),
                nullptr,
                params
            );
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;
        options.max_num_iterations = 20000;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        //std::cout << summary.FullReport() << std::endl;
        

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

            std::cout << "Fitted parameters and expanded uncertainties (2-sigma):" << std::endl;
            std::cout << "A = " << params[0] << " UA = " << 2 * std::sqrt(cov[0]) << std::endl;
            std::cout << "B = " << params[1] << " UB = " << 2 * std::sqrt(cov[4]) << std::endl;
            std::cout << "C = " << params[2] << " UC = " << 2 * std::sqrt(cov[8]) << std::endl;
            fit_params.A = params[0];
            fit_params.B = params[1];
            fit_params.C = params[2];
        }
    
        fit_params.D = 0.0;
        fit_params.UD = 0.0;

        return fit_params;
    }
        
    // Get median of T5, P5 for centering/scaling
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

    //propagate T5,P5,phi,A,B,C uncertainties to tau
    static double calc_u_tau_no_D_centered (double T5, double P5, double phi, double UT5, double UP5, double Uphi, const FitParams& params, double median_T5, double median_P5) {
        /*
        * model = A * exp(B*(1/T - 1/median_T)) * (P/median_P)^C * phi
        */
        double A = params.A;
        double B = params.B;
        double C = params.C;
        double U_A = params.UA;
        double U_B = params.UB;
        double U_C = params.UC;

        double U_x1 = T5 * UT5;
        double U_x2 = P5 * UP5;
        double U_x3 = phi * Uphi;

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


    static std::vector<double> calc_random_uncertainties(const std::vector<TestTimeIFs>& test_list, double phi, double uT5, double uP5, double uphi, const std::vector<double>& systematic_relative_errors) {
        std::vector<double> random_uncertainties;
        random_uncertainties.reserve(test_list.size());

        TestTimeIFs medians = median(test_list);
        // You can set medians.T5 =std::numeric_limits<double>::max() and  medians.P5 = 1 to remove reference centering/scaling

        const FitParams fit_params = nonlinear_least_squares_centered_no_D(test_list, phi, medians.T5, medians.P5, systematic_relative_errors);

        for (const auto& entry : test_list) {
            double u_tau = calc_u_tau_no_D_centered(entry.T5, entry.P5, phi, uT5, uP5, uphi, fit_params, medians.T5, medians.P5);
            random_uncertainties.push_back(u_tau);
        }
        return random_uncertainties;
    }

    // Combine systematic and random uncertainties
    static std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
    calc_final_expanded_uncertainties(const std::vector<TestTimeIFs>& test_list,
                                      const std::vector<double>& systematic,
                                      const std::vector<double>& random)
    {
        std::vector<double> combined(systematic.size());
        std::vector<double> final_expanded(systematic.size());
        std::vector<double> percent(systematic.size());

        for (size_t i = 0; i < systematic.size(); ++i) {
            combined[i] = systematic[i] + random[i];
            final_expanded[i] = std::sqrt(combined[i]) * 2; // k=2 for 95% confidence interval
            percent[i] = final_expanded[i] / test_list[i].IDT * 100.0;
        }
        return {combined, final_expanded, percent};
    }
};

int main(int argc, char* argv[]) { 

    std::string input_path;
    std::string output_path = "UQ_results.csv";

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





