#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>


constexpr double delta_0 = 5.0;
constexpr double binary_IF_penalty = 0.37; // Penalty for boolean IFs
constexpr double phi = 3.0; // Stoichiometric ratio

// edit mu_actual_fixed_IF_list and mu_actual_test_time_IF_list with the experiment data


// Fixed IFs: 1-6,10-13
struct FixedIFs
{
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

// Test Time IFs: 7-9
struct TestTimeIFs {
    double test_time;                   // Exp. data in microseconds
    double T5;                          // Temperature in Kelvin
    double P5;                          // Pressure in atm
};


FixedIFs mu_ideal_fixed_IFs = {
    3.0,                                // Driver length [m]
    8.0,                                // Driven length [m]
    10.0,                               // Driven inner diameter [cm]
    true,                               // Tailoring
    true,                               // Inserts
    true,                               // CRV
    true,                               // Dilution
    false,                              // Impurities
    2.0,                                // Measurement location [cm]
    true                                // Fuel-air ratio exceeding 0.3
};

FixedIFs mu_actual_fixed_IFs = {
    2.74,                               // Driver length [m]
    2.74 + 0.9144,                      // Driven length [m]
    5.08,                               // Driven inner diameter [cm]
    true,                               // Tailoring
    true,                               // Inserts
    false,                              // CRV
    true,                               // Dilution
    false,                              // Impurities
    2.0,                                // Measurement location [cm]
    true                                // Fuel-air ratio exceeding 0.3
};

std::vector<TestTimeIFs> mu_ideal_test_time_IF_list = {
    {50.0, 1000.0, 5.0 },
    {500.0, 1600.0, 20.0}
};

std::vector<TestTimeIFs> mu_actual_test_time_IF_list = {
    {4400, 1094, 2.13},
    {2100, 1106, 2.13},
    {1590, 1121, 2.13},
    {1740, 1144, 2.63},
    {1020, 1174, 2.53},
    {857,  1176, 2.13},
    {810,  1200, 2.23},
    {760,  1210, 2.23},
    {640,  1221, 2.23},
    {605,  1240, 2.53},
    {345,  1269, 2.13},
    {282,  1269, 2.13},
    {316,  1276, 2.13},
    {282,  1314, 2.53},
    {107,  1353, 2.03},
    {175,  1368, 2.43},
    {67,   1419, 2.13},
    {80,   1419, 2.03},
    {67,   1419, 1.93},
    {62,   1433, 2.03},
    {67,   1442, 2.03},
    {75,   1467, 2.33},
    {37,   1506, 2.13},
    {54,   1539, 2.43}
};


double harrington_desirability(double mu_ideal, double mu_actual) {
    double x_i = (mu_ideal - mu_actual) / mu_ideal;
    double xi = 1.0 - std::exp(-(x_i * x_i));
    double delta_i = delta_0 * xi;
    return delta_i;
}

double harrington_desirability_bool(bool mu_ideal, bool mu_actual) {
    double xi = (mu_ideal == mu_actual) ? 0.0 : binary_IF_penalty;
    double delta_i = delta_0 * xi;
    return delta_i;
}


double calc_total_fixed_harrington(const FixedIFs& mu_ideal, const FixedIFs& mu_actual) {
    double total = 0.0;

    total += harrington_desirability(mu_ideal.driver_length_m, mu_actual.driver_length_m);
    total += harrington_desirability(mu_ideal.driven_length_m, mu_actual.driven_length_m);
    total += harrington_desirability(mu_ideal.driven_inner_diameter_cm, mu_actual.driven_inner_diameter_cm);
    total += harrington_desirability(mu_ideal.measurement_location_cm, mu_actual.measurement_location_cm);
    total += harrington_desirability_bool(mu_ideal.tailoring, mu_actual.tailoring);
    total += harrington_desirability_bool(mu_ideal.inserts, mu_actual.inserts);
    total += harrington_desirability_bool(mu_ideal.crv, mu_actual.crv);
    total += harrington_desirability_bool(mu_ideal.dilution, mu_actual.dilution);
    total += harrington_desirability_bool(mu_ideal.impurities, mu_actual.impurities);
    total += harrington_desirability_bool(mu_ideal.fuel_air_ratio, mu_actual.fuel_air_ratio);

    return total;
}

std::vector<double> calc_systematic_uncertainties(const std::vector<TestTimeIFs>& test_list, const std::vector<TestTimeIFs>& ideal_bounds, double fixed_IF_error, double delta_0){

    std::vector<double> systematic_uncertainties;

    double lower_bound = ideal_bounds.front().test_time;
    double upper_bound = ideal_bounds.back().test_time;

    for (const auto& entry : test_list) {
        double mu_actual = entry.test_time;
        double delta_val = 0.0;

        if (mu_actual < lower_bound) {
            delta_val = harrington_desirability(lower_bound, mu_actual);
        }
        else if (mu_actual > upper_bound) {
            delta_val = harrington_desirability(upper_bound, mu_actual);
        }
    
        double systematic_error = delta_val + fixed_IF_error + delta_0;
        double systematic_uncertainty = std::pow((entry.test_time * systematic_error / 100.0 / 2.0), 2);
        systematic_uncertainties.push_back(systematic_uncertainty);
    }

    return systematic_uncertainties;
}




// Random uncertainty calculations
struct Experiment {
    double T5;       // Temperature (K)
    double P5;       // Pressure (atm)
    double phi;     // Stoichiometric ratio
    double uT5;      // Uncertainty percentage of T
    double uP5;      // Uncertainty percentage of P
    double uphi;    // Uncertainty percentage of phi
};

// Fit parameters with uncertainties
struct FitParams {
    double A, B, C, D;     // regression coefficients
    double UA, UB, UC, UD; // uncertainties of coefficients
};


// Function to calculate u_c^2(y)
double calc_u_tau(const Experiment& exp, const FitParams& params) {
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

    double U_x1 = x1*u1;
    double U_x2 = x2*u2;
    double U_x3 = x3*u3;

    double usqr_x1 = (U_x1/2) * (U_x1/2);
    double usqr_x2 = (U_x2/2) * (U_x2/2);
    double usqr_x3 = (U_x3/2) * (U_x3/2);
    double usqr_xA = (U_A/2) * (U_A/2);
    double usqr_xB = (U_B/2) * (U_B/2);
    double usqr_xC = (U_C/2) * (U_C/2);
    double usqr_xD = (U_D/2) * (U_D/2);

    double exp_term = std::exp(B / x1);
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


std::vector<double> calc_random_uncertainties(const std::vector<TestTimeIFs>& test_list){

    std::vector<double> random_uncertainties;


    for (const auto& entry : test_list) {

        Experiment exp = {entry.T5, entry.P5, phi, 0.01, 0.015, 0.005}; // T,P,phi,uT,uP,uphi  
        FitParams fp  = {0.0075, 16400.0, 1.91, -3.4, 0.000046, 86.0, 0.071, 0.052 };

        double u_tau = calc_u_tau(exp, fp);
        random_uncertainties.push_back(u_tau);
    }

    return random_uncertainties;
}

struct UncertaintyQuantification {
    std::vector<double> systematic_uncertainty;
    std::vector<double> random_uncertainty;
    std::vector<double> combined_uncertainty;
    std::vector<double> final_extended_uncertainty;
    std::vector<double> percent_uncertainty;
};

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> calc_final_extended_uncertainties(const std::vector<TestTimeIFs>& test_list, const std::vector<double>& systematic, const std::vector<double>& random){
    std::vector<double> combined_uncertainties(systematic.size());
    std::vector<double> final_extended_uncertainties(systematic.size());
    std::vector<double> percent_uncertainties(systematic.size());
    for (size_t i = 0; i < systematic.size(); ++i) {
        combined_uncertainties[i] = systematic[i] + random[i];
        final_extended_uncertainties[i] = std::sqrt(combined_uncertainties[i]) * 2;
        percent_uncertainties[i] = final_extended_uncertainties[i] / test_list[i].test_time * 100.0;
    }
    return {combined_uncertainties, final_extended_uncertainties, percent_uncertainties};
}


int main()
{
    double fixed_IF_error = calc_total_fixed_harrington(mu_ideal_fixed_IFs, mu_actual_fixed_IFs);
    std::cout << "Total Harrington Delta (Fixed IFs): " << fixed_IF_error << std::endl;
    std::vector<double> systematic_uncertainties = calc_systematic_uncertainties(mu_actual_test_time_IF_list, mu_ideal_test_time_IF_list, fixed_IF_error, delta_0);
    std::cout << "\nsystematic Errors for each test time:\n";
    for (size_t i = 0; i < systematic_uncertainties.size(); ++i) {
        std::cout << "Test time: " << mu_actual_test_time_IF_list[i].test_time
                  << " -> systematic Uncertainties: " << systematic_uncertainties[i] << "\n";
    }
    std::vector<double> random_uncertainties = calc_random_uncertainties(mu_actual_test_time_IF_list);
    for (size_t i = 0; i < random_uncertainties.size(); ++i) {
        std::cout << "T5    : " << mu_actual_test_time_IF_list[i].T5
                  << " -> Random Uncertainties: " << random_uncertainties[i] << "\n";
    }
    auto [combined_uncertainties, final_extended_uncertainties, percent_uncertainties] = calc_final_extended_uncertainties(mu_actual_test_time_IF_list, systematic_uncertainties, random_uncertainties);
    std::cout << "\nTotal Errors, Uncertainties, and Percent Uncertainties for each test time:\n";
    for (size_t i = 0; i < combined_uncertainties.size(); ++i) {
        std::cout << "Test time: " << mu_actual_test_time_IF_list[i].test_time
                  << " -> Combined Uncertainties: " << combined_uncertainties[i]
                  << ", Final Extended Uncertainties: " << final_extended_uncertainties[i]
                  << ", Percent Uncertainties: " << percent_uncertainties[i] << "%\n";
    }

    return 0;
}