import numpy as np
from scipy import integrate
import EvolutionaryAlgorithm


def MarkovStep(U0, U1, WB, phiC31, ks):
    """Runs recombinase model using a Markov Chain algorithm"""
    t = 0.0
    U_WB, L_WB, R_WB, B_WB = U0
    U_phiC31, L_phiC31, R_phiC31, B_phiC31 = U1
    k1f_WB, k1r_WB, k2f_WB, k1f_phiC31, k1r_phiC31, k2f_phiC31 = ks
    # Creating Transition Matrices:
    transition_matrix_WB = np.zeros((4, 4))
    transition_matrix_phiC31 = np.zeros((4, 4))
    transition_matrix_WB[0, 1:3] = k1f_WB * WB
    transition_matrix_WB[0, 0] = 1 - 2 * k1f_WB * WB
    transition_matrix_WB[1, 0] = k1r_WB
    transition_matrix_WB[2, 0] = k1r_WB
    transition_matrix_WB[1, -1] = k2f_WB * WB * (1-B_phiC31)
    transition_matrix_WB[2, -1] = k2f_WB * WB * (1-B_phiC31)
    transition_matrix_WB[1, 0] = k1r_WB
    transition_matrix_WB[2, 0] = k1r_WB
    transition_matrix_WB[1, 1] = 1 - (k1r_WB + k2f_WB * WB)
    transition_matrix_WB[2, 2] = 1 - (k1r_WB + k2f_WB * WB)
    transition_matrix_WB[-1, -1] = 1

    transition_matrix_phiC31[0, 1:3] = k1f_phiC31 * phiC31
    transition_matrix_phiC31[0, 0] = 1 - 2 * k1f_phiC31 * phiC31
    transition_matrix_phiC31[1, 0] = k1r_phiC31
    transition_matrix_phiC31[2, 0] = k1r_phiC31
    transition_matrix_phiC31[1, 0] = k1r_phiC31
    transition_matrix_phiC31[2, 0] = k1r_phiC31
    transition_matrix_phiC31[1, -1] = k2f_phiC31 * phiC31 * (1-B_WB)
    transition_matrix_phiC31[2, -1] = k2f_phiC31 * phiC31 * (1-B_WB)
    transition_matrix_phiC31[1, 1] = 1 - (k1r_phiC31 + k2f_phiC31 * phiC31)
    transition_matrix_phiC31[2, 2] = 1 - (k1r_phiC31 + k2f_phiC31 * phiC31)
    transition_matrix_phiC31[-1, -1] = 1

    transition_matrix_WB[transition_matrix_WB < 0] = 0
    transition_matrix_phiC31[transition_matrix_phiC31 < 0] = 0
    transition_matrix_WB[transition_matrix_WB > 1] = 1
    transition_matrix_phiC31[transition_matrix_phiC31 > 1] = 1
    WB_next_transition = np.dot(transition_matrix_WB.T, U0)
    phiC31_next_transition = np.dot(transition_matrix_phiC31.T, U1)
    WB_next = np.zeros(4)
    WB_next[np.random.choice(4, p=WB_next_transition)] = 1
    phiC31_next = np.zeros(4)
    phiC31_next[np.random.choice(4, p=phiC31_next_transition)] = 1
    return WB_next, phiC31_next


def dudt(U, dt, dox, aba, k):
    WB, phiC31 = U
    WBkf, WBkr, phiC31kf, phiC31kr = k
    dWB_dt = 0.01 + WBkf * dox ** 2 - WBkr * WB
    dphiC31_dt = 0.01 + phiC31kf * aba ** 2 - phiC31kr * phiC31
    return [dWB_dt, dphiC31_dt]


def Run(dox, aba, binding_constants, tMax=96., dt=1.):
    t = 0.
    eps = 1.0e-6
    U0, U1 = [1, 0, 0, 0], [1, 0, 0, 0]
    WB = 0.
    phiC31 = 0.
    ode_times = np.arange(t, tMax + eps, dt)
    recombinase_binding_constants = binding_constants[4:]
    dox_aba_binding_constants = binding_constants[:4]
    times = [t]
    states_WB = [U0]
    states_phiC31 = [U1]
    recombinase = integrate.odeint(dudt, [WB, phiC31], ode_times, args=(dox, aba, dox_aba_binding_constants))
    max_recombinase = integrate.odeint(dudt, [WB, phiC31], ode_times, args=(4.51, 150, dox_aba_binding_constants))
    WB_prot = recombinase[:, 0] / np.amax(max_recombinase[:, 0])
    phiC31_prot = recombinase[:, 1] / np.amax(max_recombinase[:, 0])
    while t + dt < tMax:
        states_WB.append(U0)
        states_phiC31.append(U1)
        times.append(t)
        U0, U1 = MarkovStep(U0, U1, WB, phiC31, recombinase_binding_constants)
        WB, phiC31 = WB_prot[int(t)], phiC31_prot[int(t)]
        if U0[-1] + U1[-1] > 0:
            states_WB.append(U0)
            states_phiC31.append(U1)
            times.append(t)
            if U0[-1] == 1:
                return times, states_WB, states_phiC31, 1
            else:
                return times, states_WB, states_phiC31, 2
        states_WB.append(U0)
        states_phiC31.append(U1)
        times.append(t)
        t = t + dt
    return times, states_WB, states_phiC31, 0

def RunAllTransitions(numCells, dox, aba, binding_constants, tMax=96., dt=1.):
    final_states = []
    times_all = []
    for i in range(numCells):
        times, traj_WB, traj_phiC31, final_state = Run(dox, aba, binding_constants, tMax, dt)
        final_states.append(final_state)
        times_all.append(times)
    return final_states, times_all

def RunAllInducers(inducers, k0, k1, k2, k3, k4, k5, k6, k7, k8, k9):
    binding_constants = [k0, k1, k2, k3, k4, k5, k6, k7, k8, k9]
    dox, aba = inducers
    final_rates = np.zeros((len(dox), 2))
    for i in range(len(dox)):
        final_states, transition_times = RunAllTransitions(1000, dox[i], aba[i], binding_constants)
        final_red, final_yellow = FinalTransitionRate(final_states, transition_times, T=96, dt=1.)
        final_rates[i, :] = [final_red[-1], final_yellow[-1]]
    return final_rates


def CountCumulativeTransitions(traj, transition_time, T, dt):
    eps = 1.0e-06
    times = np.arange(0, T + eps, dt)
    red_transition_time = np.zeros(len(times))
    yellow_transition_time = np.zeros(len(times))
    for i in range(len(traj)):
        if traj[i] == 1:
            red_transition_time[int(transition_time[i][-1])] += 1
        elif traj[i] == 2:
            yellow_transition_time[int(transition_time[i][-1])] += 1

    cred = np.zeros(len(times))
    cyellow = np.zeros(len(times))
    for j in range(len(times)):
        cred[j] = np.sum(red_transition_time[:j + 1])
        cyellow[j] = np.sum(yellow_transition_time[:j + 1])
    return cred, cyellow, times, len(traj)

def FinalTransitionRate(traj, transition_times, T, dt):
    cred, cyellow, times, num_trajectories = CountCumulativeTransitions(traj, transition_times, T, dt)
    return cred/num_trajectories, cyellow/num_trajectories

def PlotCumulativeTransitions(traj, transition_times, T, dt, plot=True):
    cred, cyellow, times, num_trajectories = CountCumulativeTransitions(traj, transition_times, T, dt)
    import pylab
    pylab.figure(1)
    pylab.title("Number of Transitions")
    pylab.plot(times, cred, 'r-')
    pylab.plot(times, cyellow, 'y-')
    pylab.ylim(0, len(traj))
    pylab.legend(['R+', 'Y+'])
    pylab.show()

if __name__ == '__main__':
    dox = 4.5 * (10 ** -6)
    aba = 3 * (10 ** -6)
    ks = EvolutionaryAlgorithm.generate_parameters_markov_parent(dox, aba)
    ks = [9734298.937655011, 1.5362719379096252, 9734298.937655011, 1.5362719379096252, 246.91358024691354, 0.287475612817039, 246.91358024691354, 0.22222222222222227, 0.21036577634209352, 0.22222222222222227]
    final_states, transition_times = RunAllTransitions(1000, dox, aba, ks)
    PlotCumulativeTransitions(final_states, transition_times, T=96., dt=1.)
    print(FinalTransitionRate(final_states, transition_times, T=96., dt=1.))


