#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <fstream>
#include <iomanip>
#include <locale>
#include <string>
#include <nlohmann/json.hpp>



using json = nlohmann::json;

class UncertaintyQuantification {
public:
    // ---------- Public data types ----------
    struct FixedIFs {
        double driver_length_m;             // Driver length [m]
        double driven_length_m;             // Driven length [m]
        double driven_inner_diameter_cm;    // Driven inner diameter [cm]
        bool tailoring;                     // Tailoring
        bool inserts;                       // Inserts
        bool crv;                           // CRV
        bool dilution;                      // Dilution
        bool impurities;                    // Impurities
        double measurement_location_cm;     // Measurement location [cm]
        bool fuel_air_ratio;                // Fuel-air ratio exceeding 0.3
    };

    struct TestTimeIFs {
        double test_time;   // Exp. data in microseconds
        double T5;          // Temperature in Kelvin
        double P5;          // Pressure in atm
    };

    struct ExperimentParams {
        double T5;    // Temperature (K)
        double P5;    // Pressure (atm)
        double phi;   // Stoichiometric ratio
        double uT5;   // Uncertainty (fraction) of T
        double uP5;   // Uncertainty (fraction) of P
        double uphi;  // Uncertainty (fraction) of phi
    };

    struct FitParams {
        double A, B, C, D;     // regression coefficients
        double UA, UB, UC, UD; // uncertainties of coefficients
    };

    struct Experiment {
        FixedIFs fixed_IFs;    // Fixed IFs for the experiment
        std::vector<TestTimeIFs> test_time_IF_list;  // Test time IFs for the experiment
        double phi;            // Stoichiometric ratio
        std::string mixture;   // Mixture description
    };

    struct Results {
        std::vector<double> systematic_uncertainty;
        std::vector<double> random_uncertainty;
        std::vector<double> combined_uncertainty;
        std::vector<double> final_extended_uncertainty;
        std::vector<double> percent_uncertainty;
    };

    // ---------- Constructors ----------
    UncertaintyQuantification(const std::string& filepath) 
        : experiments_(readJSON(filepath)) {
        
        
        mu_ideal_fixed_IFs_ = {
            3.0,   8.0, 10.0,
            true,  true, true,  true,  false,
            2.0,   true
        };

        mu_ideal_test_time_IF_list_ = {
            {50.0, 1000.0, 5.0},
            {500.0, 1600.0, 20.0}
        };

    }




    Results compute(const Experiment& exp) const {
        Results out;

        out.systematic_uncertainty =
            calc_systematic_uncertainties(mu_ideal_fixed_IFs_, 
                                          exp.fixed_IFs,
                                          exp.test_time_IF_list,
                                          mu_ideal_test_time_IF_list_);

        out.random_uncertainty =
            calc_random_uncertainties(exp.test_time_IF_list, exp.phi);

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
        // Compute results
        auto res = compute(experiments_[2]);
        const auto& tests = experiments_[2].test_time_IF_list;  // uses your current data

        if (tests.size() != res.percent_uncertainty.size()) return false;

        std::ofstream out(filepath);
        if (!out) return false;

        // Ensure decimal point is '.'
        out.imbue(std::locale::classic());
        out << std::fixed << std::setprecision(2);

        // Header
        out << "ignition_delay_time,T5_K,P5_atm,"
            "systematic,random,combined,final_extended,percent\n";

        // Rows
        for (size_t i = 0; i < tests.size(); ++i) {
            out << tests[i].test_time << ','
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

    std::vector<Experiment> readJSON(const std::string& filepath) {
        
        std::vector<Experiment> experiments{};
        // Load JSON from file
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Could not open " << filepath << "\n";
            return {};
        }

        json j;
        file >> j;

        // Access Experiments
        for (const auto& exp : j["Experiments"]) {
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

            for (const auto& entry : exp["TestTimeIFsList"]) {
                TestTimeIFs test_time_IF;
                test_time_IF.test_time = entry["test_time"];
                test_time_IF.T5 = entry["T5"];
                test_time_IF.P5 = entry["P5"];
                experiment.test_time_IF_list.push_back(test_time_IF);
            }

            experiment.phi = exp["phi"];
            experiment.mixture = exp["mixture"];

            experiments.push_back(experiment);
        }
        return experiments;

    }

private:
    // ---------- Constants (configurable via setters) ----------
    double delta_0_ = 5.0;
    double binary_IF_penalty_ = 0.37;  // Penalty for boolean IFs

    // Default experiment/fit uncertainties
    double uT5_  = 0.01;   // 1%
    double uP5_  = 0.015;  // 1.5%
    double uphi_ = 0.005;  // 0.5%

    // FitParams fit_params_ { 0.0075, 16400.0, 1.91, -3.4,  // nadia
    //                         0.000046, 86.0, 0.071, 0.052 };
    //FitParams fit_params_ { 0.688794, 9266.79, -0.684123, 0,  // davidson
                            // 0.09, 200.0, 0.036, 0 };
    FitParams fit_params_ { 0.0233594, 15633.9, -1.08387, 0,  //shen
                             0.0034, 190, 0.018, 0 };
    

    // ---------- Stored inputs ----------
    FixedIFs mu_ideal_fixed_IFs_{};
    std::vector<TestTimeIFs> mu_ideal_test_time_IF_list_{};
    std::vector<Experiment> experiments_{};


    // ---------- Core math ----------
    double harrington_desirability(double mu_ideal, double mu_actual) const {
        double x_i = (mu_ideal - mu_actual) / mu_ideal;
        double xi = 1.0 - std::exp(-(x_i * x_i));
        double delta_i = delta_0_ * xi;
        return delta_i;
    }

    double harrington_desirability_bool(bool mu_ideal, bool mu_actual) const {
        double xi = (mu_ideal == mu_actual) ? 0.0 : binary_IF_penalty_;
        double delta_i = delta_0_ * xi;
        return delta_i;
    }

    double calc_total_fixed_harrington(const FixedIFs& mu_ideal, const FixedIFs& mu_actual) const {
        double total = 0.0;
        total += harrington_desirability(mu_ideal.driver_length_m,           mu_actual.driver_length_m);
        total += harrington_desirability(mu_ideal.driven_length_m,           mu_actual.driven_length_m);
        total += harrington_desirability(mu_ideal.driven_inner_diameter_cm,  mu_actual.driven_inner_diameter_cm);
        total += harrington_desirability(mu_ideal.measurement_location_cm,   mu_actual.measurement_location_cm);
        total += harrington_desirability_bool(mu_ideal.tailoring,            mu_actual.tailoring);
        total += harrington_desirability_bool(mu_ideal.inserts,              mu_actual.inserts);
        total += harrington_desirability_bool(mu_ideal.crv,                  mu_actual.crv);
        total += harrington_desirability_bool(mu_ideal.dilution,             mu_actual.dilution);
        total += harrington_desirability_bool(mu_ideal.impurities,           mu_actual.impurities);
        total += harrington_desirability_bool(mu_ideal.fuel_air_ratio,       mu_actual.fuel_air_ratio);
        return total;
    }

    std::vector<double> calc_systematic_uncertainties(const FixedIFs& ideal_fixed, 
                                                      const FixedIFs& actual_fixed,
                                                      const std::vector<TestTimeIFs>& test_list,
                                                      const std::vector<TestTimeIFs>& ideal_bounds) const
    {
        double fixed_IF_error = calc_total_fixed_harrington(ideal_fixed, actual_fixed);

        std::vector<double> systematic_uncertainties;
        systematic_uncertainties.reserve(test_list.size());

        double lower_bound = ideal_bounds.front().test_time;
        double upper_bound = ideal_bounds.back().test_time;

        for (const auto& entry : test_list) {
            double mu_actual = entry.test_time;
            double delta_val = 0.0;

            if (mu_actual < lower_bound) {
                delta_val = harrington_desirability(lower_bound, mu_actual);
            } else if (mu_actual > upper_bound) {
                delta_val = harrington_desirability(upper_bound, mu_actual);
            }

            double systematic_error = delta_val + fixed_IF_error + delta_0_;
            double systematic_uncertainty = std::pow((entry.test_time * systematic_error / 100.0 / 2.0), 2);
            systematic_uncertainties.push_back(systematic_uncertainty);
        }

        return systematic_uncertainties;
    }

    static double calc_u_tau(const ExperimentParams& exp, const FitParams& params) {
        double x1 = exp.T5;
        double x2 = exp.P5;
        double x3 = exp.phi;

        double u1 = exp.uT5;
        double u2 = exp.uP5;
        double u3 = exp.uphi;

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

        // std::cout << "Calculating u_tau for T5=" << x1 << ", P5=" << x2 << ", phi=" << x3 << "\n";
        // std::cout << "Intermediate terms: exp_term=" << exp_term << ", base_term=" << base_term << "\n";

        double q1 = std::pow(A * base_term * (-B / (x1 * x1)), 2) * usqr_x1;
        double q2 = std::pow(C * A * exp_term * std::pow(x2, C - 1) * std::pow(x3, D), 2) * usqr_x2;
        double q3 = std::pow(D * A * exp_term * std::pow(x2, C) * std::pow(x3, D - 1), 2) * usqr_x3;
        double q4 = std::pow(exp_term * std::pow(x2, C) * std::pow(x3, D), 2) * usqr_xA;
        double q5 = std::pow((1.0 / x1) * A * exp_term * std::pow(x2, C) * std::pow(x3, D), 2) * usqr_xB;
        double q6 = std::pow(A * exp_term * std::pow(x2, C) * std::pow(x3, D) * std::log(x2), 2) * usqr_xC;
        double q7 = std::pow(A * exp_term * std::pow(x2, C) * std::pow(x3, D) * std::log(x3), 2) * usqr_xD;

        // std::cout << "u_tau components: " << q1 << ", " << q2 << ", " << q3 << ", " << q4 << ", " << q5 << ", " << q6 << ", " << q7 << "\n";

        return q1 + q2 + q3 + q4 + q5 + q6 + q7;
    }

    std::vector<double> calc_random_uncertainties(const std::vector<TestTimeIFs>& test_list, double phi) const {
        std::vector<double> random_uncertainties;
        random_uncertainties.reserve(test_list.size());

        for (const auto& entry : test_list) {
            ExperimentParams exp { entry.T5, entry.P5, phi, uT5_, uP5_, uphi_ };
            double u_tau = calc_u_tau(exp, fit_params_);
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
            final_extended[i] = std::sqrt(combined[i]) * 2;
            percent[i] = final_extended[i] / test_list[i].test_time * 100.0;
        }
        return {combined, final_extended, percent};
    }
};



int main() {
    std::string filename = "shen_2009"; // Change to "nadia" or "davidson" as needed
    UncertaintyQuantification uq(std::string(PROJECT_ROOT) + "/data/" + filename + ".json");
    
    if (!uq.writeCSV(std::string(PROJECT_ROOT) + "/data/"+ filename + "_UQ_results.csv")) {
        std::cerr << "CSV write failed.\n";
        return 1;
    }
    std::cout << "Wrote " << filename << "_UQ_results.csv\n";
    // uq.run(); // prints the same outputs as your original main()
}





