import pandas as pd
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
from scipy.stats import qmc
import time


mechanism_path= "C:\\Users\\barki\\Desktop\\master thesis\\papers\\mechanism\\ScienceDirect_files_18Dec2025_16-58-12.343\\C2H4_2021.yaml"

rxns = [
    {"equation": "C2H4 + O <=> CH3 + HCO", "f_range": [0.135, 0.142]},
    {"equation": "C2H4 + O <=> CH2CO + H2", "f_range": [0.135, 0.142]},
    {"equation": "C2H4 + O <=> CH2CHO + H", "f_range": [0.135, 0.142]},
    {"equation": "C2H4 + OH <=> C2H3 + H2O", "f_range": [0.301, 0.318]},
    {"equation": "C2H4 + OH <=> CH2CH2OH", "f_range": [0.143, 0.149]},
    {"equation": "C2H4 + H (+M) <=> C2H5 (+M)", "f_range": [0.22, 0.22]},
    {"equation": "C2H4 + H <=> C2H3 + H2", "f_range": [0.847, 0.898]},
    {"equation": "C2H5 + O2 <=> C2H4 + HO2", "f_range": [0.231, 0.241]},
    {"equation": "C2H5 + O2 <=> C2H5O2", "f_range": [0.491, 0.511]}
]


def lhs_sampling(rxns):
    for reaction in rxns:
        reaction["f_value"] = np.mean(reaction["f_range"])
    print(rxns)
    factors_list = [np.mean(rxn["f_range"]) for rxn in rxns]
    dim = len(factors_list)

    sampler = qmc.LatinHypercube(dim)
    sample = sampler.random(n=1000)
    l_bounds = 1/np.pow(10, factors_list)
    u_bounds = 1*np.pow(10, factors_list)
    qmc.scale(sample, l_bounds, u_bounds)
    return sample

def find_reaction_index_by_equation(gas, equation: str) -> int:
    for i, rxn in enumerate(gas.reactions()):
        if rxn.equation == equation:
            return i
    raise ValueError(f"Reaction not found (exact match): {equation}")

def get_A(ct_rxn):
    if ct_rxn.reaction_type == 'Arrhenius' or ct_rxn.reaction_type == 'three-body-Arrhenius':
        A = [ct_rxn.rate.pre_exponential_factor]
    elif ct_rxn.reaction_type == 'falloff-Troe' or ct_rxn.reaction_type == 'falloff-Lindemann':
        A_low = ct_rxn.rate.low_rate.pre_exponential_factor
        A_high = ct_rxn.rate.high_rate.pre_exponential_factor
        A = [A_low, A_high]
    elif ct_rxn.reaction_type == 'pressure-dependent-Arrhenius':
        A = [p[1].pre_exponential_factor for p in ct_rxn.rate.rates]
    else:
        raise ValueError(f"Reaction has unhandled reaction type: {ct_rxn.reaction_type}")
    return A
    
def multiply_A_in_dict(d, base_A, m):
    rtype = d.get("type", None)

    # Elementary (no "type") or explicit elementary
    if rtype is None or rtype == "elementary":
        d["rate-constant"]["A"] = base_A[0]* m
        return d

    # Three-body
    if rtype == "three-body":
        d["rate-constant"]["A"] = base_A[0] * m
        return d

    # Falloff (Troe or Lindemann): multiply both limits
    if rtype == "falloff":
        d["low-P-rate-constant"]["A"]  = base_A[0] * m
        d["high-P-rate-constant"]["A"] = base_A[1] * m
        return d

    # Pressure-dependent Arrhenius (PLOG)
    if rtype == "pressure-dependent-Arrhenius":
        for i, entry in enumerate(d["rate-constants"]):
            entry["A"] = base_A[i] * m
        return d

    raise NotImplementedError(f"Unsupported reaction dict type={rtype}")

def apply_multiplier_to_reaction(gas, base_A, i, m):
    old = gas.reaction(i)
    d = dict(old.input_data)

    d2 = multiply_A_in_dict(d, base_A, m)
    new = ct.Reaction.from_dict(d2, gas)
    gas.modify_reaction(i, new)
    print(f"Applied multiplier {m} to reaction {i}: {old.equation}")
    
def multiply_all_A(gas, rxns, m_sample):
    for idx, rxn in enumerate(rxns):
        apply_multiplier_to_reaction(gas, rxn["base_A"], rxn["id"], m_sample[idx])

def calculate_ignition_delay_constV_batch_reactor(gas, T5_list, P5_list, reactants, meas_IDT = None, plot=False, t_max = 0.1):
    # batch reactor const volume
    ignition_delay_times = []
    for T5, P5 in zip(T5_list, P5_list):
        
        # Define the reactor temperature and pressure
        reactor_temperature = T5  # Kelvin
        reactor_pressure_atm = P5  # atm
        reactor_pressure = reactor_pressure_atm * 101325  # Pascals

        gas.TPX = reactor_temperature, reactor_pressure, reactants

        reactor = ct.IdealGasReactor(gas, energy='on')  # constant volume if volume not changed
        reactor_network = ct.ReactorNet([reactor])
        
        time_history = ct.SolutionArray(gas, extra="t")

        t0 = time.time()
        t = 0

        counter = 1
        while t < t_max:
            t = reactor_network.step()
            if not counter % 10:
                time_history.append(reactor.thermo.state, t=t)
            counter += 1

        # Ignition delay time defined as the time corresponding to the maximum temperature rise rate
        tau = time_history.t[np.argmax(np.gradient(time_history.T, time_history.t))]
        
        ignition_delay_times.append(tau)

        t1 = time.time()

        print(f"T5 = {T5} K, P5 = {P5} atm, Computed Ignition Delay: {tau:.3e} seconds. Took {t1-t0:3.2f}s to compute")
        
    if plot:
        scaled_inverse_T5 = [1000.0 / T for T in T5_list]

        plt.figure(figsize=(8, 6))
        if meas_IDT is not None:
            plt.scatter(scaled_inverse_T5, meas_IDT, label='real experiment', color='red')
        plt.scatter(scaled_inverse_T5, ignition_delay_times, label='Cantera', color='blue')
        plt.xlabel('1000/T5 (1/K)')
        plt.ylabel('Ignition Delay Time (s)')
        plt.yscale('log')
        plt.legend()
        plt.title('Ignition Delay Times Comparison')
        plt.grid(True)
        plt.show()
        
    return ignition_delay_times




#Setup mechanism
gas = ct.Solution('C2H4_2021.yaml')
for rxn in rxns:
    rxn["id"] = find_reaction_index_by_equation(gas, rxn["equation"])
    rxn["base_A"] = get_A(gas.reaction(rxn["id"]))


#Sample preexponential factors
param_multipliers_samples = lhs_sampling(rxns)        
multipliers_sample = param_multipliers_samples[0]
print(multipliers_sample)
multiply_all_A(gas, rxns, multipliers_sample)

# Now gas has modified reaction rates according to the sampled multipliers
# now we simulate a simple ignition delay
T5_list = [1683, 1621, 1488, 1481, 1475, 1455, 1452, 1439, 1426, 1423,
    1416, 1414, 1412, 1400, 1397, 1388, 1388, 1386, 1357, 1286, 1286]
P5_list = [0.98, 1.16, 1.21, 1.08, 1.19, 1.17, 1.17, 1.14, 1.13, 1.12,
    1.13, 1.11, 1.11, 1.09, 1.10, 1.09, 1.10, 1.08, 1.03, 1.27, 1.17]
measured_IDT = [7.30e-05, 7.80e-05, 1.23e-04, 1.54e-04, 1.49e-04, 1.52e-04, 1.57e-04,
    1.80e-04, 1.85e-04, 2.04e-04, 2.01e-04, 2.17e-04, 1.96e-04, 2.27e-04,
    2.20e-04, 2.36e-04, 2.47e-04, 2.54e-04, 3.18e-04, 5.37e-04, 4.99e-04]
reactants = 'C2H4:0.01, O2:0.03, AR:0.96'
# T5_list = [
#     1228, 1276, 1295, 1331, 1341, 1388, 1416, 1468, 1487,
#     1507, 1530, 1554, 1625, 1647, 1652, 1732, 1746
# ]
# P5_list = [
#     3.26, 3.16, 2.90, 3.05, 2.91, 2.88, 2.99, 2.77, 3.02,
#     2.77, 2.96, 2.73, 2.79, 2.61, 2.72, 2.61, 2.61
# ]
# measured_IDT = [
#     6.90e-04, 3.65e-04, 3.06e-04, 2.01e-04, 1.85e-04, 1.19e-04,
#     9.23e-05, 6.69e-05, 5.63e-05, 5.29e-05, 4.41e-05,
#     4.29e-05, 3.09e-05, 3.06e-05, 2.90e-05, 2.36e-05, 2.28e-05
# ]

ignition_delay_times = calculate_ignition_delay_constV_batch_reactor(gas, T5_list, P5_list, reactants, meas_IDT=measured_IDT, plot=True)


# ok that works. NOw 
    