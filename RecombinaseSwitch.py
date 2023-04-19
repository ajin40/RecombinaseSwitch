import numpy as np
import random
from Reactions import *


class RecombinaseSwitch:
    """
    RecombinaseSwitch

    The set of Chemical states (all 3 mRNAs, all 3 proteins, and all 9
    promoter states) have a defined ordering, as stored in the
    dictionary chemIndex, which maps Chemicals to integer indices in
    the ordering.  This ordering is necessary to communicate with utilities
    (such as scipy.integrate.odeint) that require chemical amounts to
    be stored as a 1D array.  The GetStateVector method returns the
    Chemical amounts as a scipy array in the appropriate order, and
    the SetStateVector method sets the Chemical amounts based on a
    supplied array.
    """

    def __init__(self,
                 dox_conc=0,
                 aba_conc=0,
                 mRNA_degradation_rate=np.log(2.) / 120.,
                 protein_degradation_rate=np.log(2.) / 600.,
                 translation_rate=0.167,
                 occ_transcription_rate=0.5,
                 unocc_transcription_rate=5.0e-04,
                 dox_binding=1.0,
                 dox_unbinding=9.0,
                 aba_binding=1.0,
                 aba_unbinding=9.0,
                 TF_binding=1.0,
                 TF_unbinding=9.0,
                 WB_binding=0.1,
                 WB_unbinding=225.0,
                 phiC31_binding=0.1,
                 phiC31_unbinding=225.0):
        self.t = 0.
        self.chemIndex = {}
        self.reactions = []
        self.mRNA_degradation_rate = mRNA_degradation_rate
        self.protein_degradation_rate = protein_degradation_rate
        self.translation_rate = translation_rate
        self.unocc_transcription_rate = unocc_transcription_rate
        self.occ_transcription_rate = occ_transcription_rate
        self.dox_binding_rate = dox_binding
        self.dox_unbinding_rate = dox_unbinding
        self.aba_binding_rate = aba_binding
        self.aba_unbinding_rate = aba_unbinding

        self.TF_binding_rate = TF_binding
        self.TF_unbinding_rate = TF_unbinding

        self.WB_binding_rate = WB_binding
        self.WB_unbinding_rate = WB_unbinding

        self.phiC31_binding_rate = phiC31_binding
        self.phiC31_unbinding_rate = phiC31_unbinding

        self.dox = Chemical(dox_conc)
        self.aba = Chemical(aba_conc)
        self.On = Chemical(1.0)
        self.rtTA3 = Chemical(100.0)
        self.PhIF_ABI = Chemical(100.0)
        self.VPR_Pyl1 = Chemical(100.0)
        self.WB = Chemical(0.0)
        self.PhiC31 = Chemical(0.0)

        self.P_rtTA3 = Chemical(30.0)
        self.P_PhIF_ABI = Chemical(20.0)
        self.P_VPR_Pyl1 = Chemical(10.0)
        self.P_WB = Chemical(0.0)
        self.P_PhiC31 = Chemical(0.0)

        self.P_rtTA3_dox = Chemical(0.0)
        self.P_PhIF_ABI_VPR_Pyl1_aba = Chemical(0.0)

        self.UNBOUND_P_WB = Chemical(1.0)
        self.UNBOUND_P_PhiC31 = Chemical(1.0)

        self.BOUND_P_WB = Chemical(0.0)
        self.BOUND_P_PhiC31 = Chemical(0.0)

        self.UNBOUND_RED = Chemical(1.0)
        self.UNBOUND_YELLOW = Chemical(1.0)

        self.LEFTBOUND_RED = Chemical(0.0)
        self.LEFTBOUND_YELLOW = Chemical(0.0)

        self.RIGHTBOUND_RED = Chemical(0.0)
        self.RIGHTBOUND_YELLOW = Chemical(0.0)

        self.DOUBLEBOUND_RED = Chemical(0.0)
        self.DOUBLEBOUND_YELLOW = Chemical(0.0)
        self.mRNAs = [self.rtTA3,
                      self.PhIF_ABI,
                      self.VPR_Pyl1,
                      self.WB,
                      self.PhiC31]
        self.proteins = [self.P_rtTA3,
                         self.P_PhIF_ABI,
                         self.P_VPR_Pyl1,
                         self.P_WB,
                         self.P_PhiC31]
        self.activated_proteins = [self.P_rtTA3_dox,
                                   self.P_PhIF_ABI_VPR_Pyl1_aba]
        self.P0 = [self.UNBOUND_P_WB,
                   self.UNBOUND_P_PhiC31]
        self.P1 = [self.BOUND_P_WB,
                   self.BOUND_P_PhiC31]
        self.inducers = [self.dox,
                         self.aba]
        self.S0 = [self.UNBOUND_RED,
                   self.UNBOUND_YELLOW]
        self.S1 = [self.LEFTBOUND_RED,
                   self.LEFTBOUND_YELLOW,
                   self.RIGHTBOUND_RED,
                   self.RIGHTBOUND_YELLOW]
        self.S2 = [self.DOUBLEBOUND_RED,
                   self.DOUBLEBOUND_YELLOW]
        for chem in self.inducers:
            self.AddChemical(chem)
        for chem in self.mRNAs:
            self.AddChemical(chem)
        self.AddChemical(self.On)
        for chem in self.proteins:
            self.AddChemical(chem)
        for chem in self.P0:
            self.AddChemical(chem)
        for chem in self.P1:
            self.AddChemical(chem)
        for chem in self.S0:
            self.AddChemical(chem)
        for chem in self.S1:
            self.AddChemical(chem)
        for chem in self.S2:
            self.AddChemical(chem)

        # Adding Every Reaction:
        # mRNA Degradations ---------------------------------------------
        for i in range(len(self.mRNAs)):
            self.AddReaction(
                DegradationReaction(self.mRNAs[i], mRNA_degradation_rate)
            )
        # Protein Degradations ------------------------------------------
        for i in range(len(self.proteins)):
            self.AddReaction(
                DegradationReaction(self.proteins[i], mRNA_degradation_rate)
            )
        # mRNA Productions ----------------------------------------------
        for i in range(len(self.mRNAs)):
            # rtTA3, PhIF-ABI, and WPI-Pyl1 are produced at a constant rate
            if i < 3:
                self.AddReaction(
                    CatalyzedSynthesisReaction(self.On,
                                               self.mRNAs[i],
                                               occ_transcription_rate)
                )
            # WB and PhiC31 are produced at a rate equal to bound TFs
            # P_rtTA3_dox and P_PhIF_ABI_VPR_Pyl1_aba respectively
            else:
                self.AddReaction(
                    CatalyzedSynthesisReaction(self.P1[i - 3],
                                               self.mRNAs[i],
                                               occ_transcription_rate)
                )
                self.AddReaction(
                    CatalyzedSynthesisReaction(self.P0[i - 3],
                                               self.mRNAs[i],
                                               unocc_transcription_rate)
                )
        # Protein Productions ------------------------------------------
        for i in range(len(self.proteins)):
            self.AddReaction(
                CatalyzedSynthesisReaction(self.mRNAs[i],
                                           self.proteins[i],
                                           translation_rate)
            )
        # Binding Reactions ---------------------------------------------
        # P_rtTA3 + dox -> P_rtTA3_dox
        self.AddReaction(
            HeterodimerDoubleBindingReaction(self.P_rtTA3,
                                       self.dox,
                                       self.P_rtTA3_dox,
                                       self.dox_binding_rate)
        )
        # P_rtTA3_dox -> P_rtTA3 + dox
        self.AddReaction(
            HeterodimerDoubleUnbindingReaction(self.P_rtTA3_dox,
                                         self.P_rtTA3,
                                         self.dox,
                                         self.dox_unbinding_rate)
        )
        # P_PhIF_ABI + P_VPR_Pyl1 + aba -> P_PhIF_ABI_VPR_Pyl1_aba
        self.AddReaction(
            TrimerBindingReaction(self.P_PhIF_ABI,
                                  self.P_VPR_Pyl1,
                                  self.aba,
                                  self.P_PhIF_ABI_VPR_Pyl1_aba,
                                  self.aba_binding_rate)
        )
        # P_PhIF_ABI_VPR_Pyl1_aba -> P_PhIF_ABI + P_VPR_Pyl1 + aba
        self.AddReaction(
            TrimerUnbindingReaction(self.P_PhIF_ABI_VPR_Pyl1_aba,
                                    self.P_PhIF_ABI,
                                    self.P_VPR_Pyl1,
                                    self.aba,
                                    self.aba_unbinding_rate)
        )
        # P_rtTA3 + dox -> P_rtTA3_dox
        self.AddReaction(
            HeterodimerDoubleBindingReaction(self.P_rtTA3,
                                             self.dox,
                                             self.P_rtTA3_dox,
                                             self.dox_binding_rate)
        )
        # P_rtTA3_dox + P0 -> P1
        self.AddReaction(
            HeterodimerBindingReaction(self.P_rtTA3_dox,
                                       self.P0[0],
                                       self.P1[0],
                                       self.TF_binding_rate)
        )
        # P_PhIF_ABI_VPR_Pyl1_aba + P0 -> P1
        self.AddReaction(
            HeterodimerBindingReaction(self.P_PhIF_ABI_VPR_Pyl1_aba,
                                       self.P0[1],
                                       self.P1[1],
                                       self.TF_binding_rate)
        )

        # P1 -> P_rtTA3_dox + P0
        self.AddReaction(
            HeterodimerUnbindingReaction(self.P1[0],
                                         self.P_rtTA3_dox,
                                         self.P0[0],
                                         self.TF_unbinding_rate)
        )
        # P1 -> P_PhIF_ABI_VPR_Pyl1_aba + P0
        self.AddReaction(
            HeterodimerUnbindingReaction(self.P1[1],
                                         self.P_PhIF_ABI_VPR_Pyl1_aba,
                                         self.P0[1],
                                         self.TF_unbinding_rate)
        )
        # State Change Reactions --------------------------------------
        # S0 + P_WB/P_PhiC31 -> S1 ---------------
        self.AddReaction(
            HeterodimerBindingReaction(self.S0[0],
                                       self.P_WB,
                                       self.S1[0],
                                       self.WB_binding_rate)
        )
        self.AddReaction(
            HeterodimerBindingReaction(self.S0[1],
                                       self.P_PhiC31,
                                       self.S1[1],
                                       self.phiC31_binding_rate)
        )
        # S1 -> S0 + P_WB/P_PhiC31 ----------------
        self.AddReaction(
            HeterodimerUnbindingReaction(self.S1[0],
                                         self.P_WB,
                                         self.S0[0],
                                         self.WB_unbinding_rate)
        )
        self.AddReaction(
            HeterodimerUnbindingReaction(self.S1[1],
                                         self.P_WB,
                                         self.S0[1],
                                         self.phiC31_unbinding_rate)
        )
        # S1 + P_WB/P_PhiC31-> S2 -----------------
        self.AddReaction(
            RecombinaseExcisionReaction(self.S1[0],
                                        self.P_WB,
                                        self.S2[1],
                                        self.S2[0],
                                        self.WB_binding_rate)
        )
        self.AddReaction(
            RecombinaseExcisionReaction(self.S1[1],
                                        self.P_PhiC31,
                                        self.S2[0],
                                        self.S2[1],
                                        self.phiC31_binding_rate)
        )

        self.rates = np.zeros(len(self.reactions), float)
        for rIndex, r in enumerate(self.reactions):
            self.rates[rIndex] = r.GetRate()

    def AddChemical(self, chemical):
        self.chemIndex[chemical] = len(self.chemIndex)

    def GetChemicalIndex(self, chemical):
        return self.chemIndex[chemical]

    def AddReaction(self, reaction):
        self.reactions.append(reaction)

    def GetStateVector(self):
        c = np.zeros(len(self.chemIndex), float)
        for chem, index in list(self.chemIndex.items()):
            c[index] = chem.amount
        return c

    def SetFromStateVector(self, c):
        for chem, index in list(self.chemIndex.items()):
            chem.amount = c[index]


class StochasticSwitch(RecombinaseSwitch):
    """
    StochasticSwitch is a stochastic implementation of the two chemical
    inducible recombinase system, with time evolution implemented
    using Gillespie's Direct Method, as described in detail in Step.
    """

    def ComputeReactionRates(self):
        """ComputeReactionRates computes the current rate for every
        reaction defined in the network, and stores the rates in self.rates."""
        for index, r in enumerate(self.reactions):
            self.rates[index] = r.GetRate()

    def Step(self, dtmax):
        """Step(self, dtmax) implements Gillespie's Direct Simulation Method,
        executing at most one reaction and returning the time increment
        required for that reaction to take place.  If no reaction is executed,
        the specified maximal time increment dtmax is returned.

        (1) all reaction rates are computed
        (2) a total rate for all reactions is found
        (3) a random time is selected, to be drawn from an exponential
            distribution with mean value given by the inverse of the total
            rate, e.g., ran_dtime = -scipy.log(1.-random.random())/total_rate
        (4) if the random time is greater than the time interval under
            consideration (dtmax), then no reaction is executed and dtmax
            is returned
        (5) otherwise, a reaction is chosen at random with relative
            probabilities given by the relative reaction rates;
            this is done by
            (5a) uniformly drawing a random rate from the interval from
                 [0., total rate)
            (5b) identifying which reaction rate interval corresponds to
                 the randomly drawn rate, e.g.,

                 |<-------------------total rate---------------------->|
                 |<----r0----->|<-r1->|<--r2-->|<-----r3----->|<--r4-->|
                 |                                 X                   |

                 Randomly drawn rate X lands in interval r3
        (6) the chosen reaction is executed
        (7) the time at which the reaction is executed is returned
        """
        self.ComputeReactionRates()
        total_rate = sum(self.rates)
        # get exponentially distributed time
        ran_time = -np.log(1. - random.random()) / total_rate
        if ran_time > dtmax:
            return dtmax
        # get uniformly drawn rate in interval defined by total_rate
        ran_rate = total_rate * random.random()
        # find interval corresponding to random rate
        reac_index = len(self.rates) - sum(np.cumsum(self.rates) > ran_rate)
        reaction = self.reactions[reac_index]
        # execute specified reaction
        for chem, dchem in list(reaction.stoichiometry.items()):
            chem.amount += dchem
        # return time at which reaction takes place
        return ran_time

    def Run(self, T, delta_t=0.0):
        """Run(self, T, delta_t) runs the StochasticRepressilator for
        a specified time interval T, returning the trajectory at
        specified time intervals delta_t (or, if delta_t == 0.0, after
        every reaction)
        """
        tfinal = self.t + T
        if delta_t == 0.:
            ts = [self.t]
            trajectory = [self.GetStateVector()]
            while self.S2[0].amount + self.S2[1].amount != 1:
                if self.t > tfinal:
                    return np.array(ts), np.array(trajectory)
                dt = self.Step(tfinal - self.t)
                self.t += dt
                ts.append(self.t)
                trajectory.append(self.GetStateVector())
            return np.array(ts), np.array(trajectory)
        else:
            eps = 1.0e-06
            ts = np.arange(0., T + eps, delta_t)
            trajectory = np.zeros((len(ts), len(self.chemIndex)),
                                  float)
            trajectory[0] = self.GetStateVector()
            tindex = 0
            while self.S2[0].amount + self.S2[1].amount != 1:
                if self.t > tfinal:
                    return ts, trajectory
                dt = self.Step(ts[tindex + 1] - self.t)
                self.t += dt
                if self.t >= ts[tindex + 1]:
                    tindex += 1
                    for chem, cindex in list(self.chemIndex.items()):
                        trajectory[tindex][cindex] = chem.amount
            tindex += 1
            for chem, cindex in list(self.chemIndex.items()):
                trajectory[tindex][cindex] = chem.amount
            return ts[:tindex + 1], trajectory[:tindex + 1]
