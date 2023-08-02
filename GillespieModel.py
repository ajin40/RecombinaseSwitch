from RecombinaseSwitch import *
from multiprocessing import Pool
import time

def RunStochasticSwitch(dox, aba, params, T=100., dt=1., plots=False, plotState=False):
    """RunStochasticSwitch(tmax, dt, plots=False, plotPromoter=False)
    creates and runs a StochasticSwitch for the specified time
    interval T, returning the trajectory in time increments dt,
    optionally using pylab to make plots of mRNA and protein
    amounts along the trajectory.
    """
    #sr = StochasticSwitch(dox_conc=dox, aba_conc=aba)
    sr = StochasticSwitch(dox_conc=dox,
                          aba_conc=aba,
                          dox_binding=params[0],
                          dox_unbinding=params[1],
                          aba_binding=params[2],
                          aba_unbinding=params[3],
                          TF_binding=params[4],
                          TF_unbinding=params[5],
                          WB_binding=params[6],
                          WB_unbinding=params[7],
                          phiC31_binding=params[8],
                          phiC31_unbinding=params[9]
                          )
    sts, straj = sr.Run(T, dt)
    curvetypes = ['r-', 'g-', 'b-', 'k-', 'm-']
    if plots:
        import pylab
        pylab.figure(1)
        pylab.title("RNAs")
        for i in range(5):
            pylab.plot(sts, straj[:, sr.chemIndex[sr.mRNAs[i]]], curvetypes[i])
        pylab.figure(2)
        pylab.title("Proteins")
        for i in range(5):
            pylab.plot(sts, straj[:, sr.chemIndex[sr.proteins[i]]], curvetypes[i])
    return sr, sts, straj


def RunAllTransitions(num, num_processes, dox, aba, params, T=100., dt=1.):
    """
    Stochastic Transitions for a population of cells (num). Outputs the number
    of cells that transition to the R+ or Y+ state over time.
    """
    list_args = [[dox, aba, param, T, dt] for param in params]
    with Pool(num_processes) as p:
        all_trajs = p.starmap(RunStochasticSwitch, list_args)
    return all_trajs


def PlotAllTransitions(num, num_processes, dox, aba, params, T=100., dt=1., plots=True):
    all_trajs = RunAllTransitions(num, num_processes, params, dox, aba, T, dt)
    if plots:
        import pylab
        pylab.figure(1)
        pylab.title("All Transitions")
        for i in range(num):
            red = all_trajs[i][2][-1, all_trajs[i][0].chemIndex[all_trajs[i][0].S2[0]]]
            yellow = all_trajs[i][2][-1, all_trajs[i][0].chemIndex[all_trajs[i][0].S2[1]]]
            if red > yellow:
                pylab.plot(all_trajs[i][1][-1], red, 'r.')
            elif yellow > red:
                pylab.plot(all_trajs[i][1][-1], yellow, 'y.')
    return all_trajs


def CountCumulativeTransitions(traj, T, dt):
    eps = 1.0e-06
    times = np.arange(0, T + eps, dt)
    red_transition_time = np.zeros(len(times))
    yellow_transition_time = np.zeros(len(times))
    for i in range(len(traj)):
        red = traj[i][2][-1, traj[i][0].chemIndex[traj[i][0].S2[0]]]
        yellow = traj[i][2][-1, traj[i][0].chemIndex[traj[i][0].S2[1]]]
        if red > yellow:
            red_transition_time[int(traj[i][1][-1])] += 1
        elif yellow > red:
            yellow_transition_time[int(traj[i][1][-1])] += 1

    cred = np.zeros(len(times))
    cyellow = np.zeros(len(times))
    for j in range(len(times)):
        cred[j] = np.sum(red_transition_time[:j + 1])
        cyellow[j] = np.sum(yellow_transition_time[:j + 1])
    return cred, cyellow, times, len(traj)

def FinalTransitionRate(traj, T, dt):
    cred, cyellow, times, num_trajectories = CountCumulativeTransitions(traj, T, dt)
    return cred/num_trajectories, cyellow/num_trajectories

def PlotCumulativeTransitions(traj, T, dt):
    cred, cyellow, times, num_trajectories = CountCumulativeTransitions(traj, T, dt)
    import pylab
    pylab.figure(1)
    pylab.title("Number of Transitions")
    pylab.plot(times, cred, 'r-')
    pylab.plot(times, cyellow, 'y-')
    pylab.ylim(0, len(traj))
    pylab.legend(['R+', 'Y+'])
    pylab.show()

if __name__ == "__main__":
    start_time = time.time()
    params = [[1, 9, 1, 9, 1, 9, 1, 9, 1, 9]] * 1000
    A = RunAllTransitions(1, 8, 10, 50, params, T=100., dt=1.)
    print("--- %s seconds ---" % (time.time() - start_time))
    PlotCumulativeTransitions(A, T=100., dt=1.)
    print(FinalTransitionRate(A, T=100., dt=1.))