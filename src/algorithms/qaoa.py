from collections import Counter
from scipy.optimize import minimize
import numpy as np
import random 
import time 

class QAOA:
    def __init__(self,depth,H):     # Class initialization. Arguments are "depth",
                                    # and a Diagonal Hamiltonian,"H".
        self.H = H 
        self.n = int(np.log2(int(len(self.H)))) # Calculates the number of qubits.

        #______________________________________________________________________________________________________
        self.X = self.new_mixerX()          # Executes a sequence of array manipulations to encapsulate the
                                            # effect of standard one body driver hamiltonian, \Sum \sigma_x,
                                            # acting on any state.
        #______________________________________________________________________________________________________


        self.min = min(self.H)                  # Calculates minimum of the Hamiltonain, Ground state energy.

        self.deg = len(self.H[self.H == self.min]) # Calculates the degeneracy of Ground states.
        self.p = depth                             # Standard qaoa depth written as "p".

        #______________________________________________________________________________________________________

                    # The sequence of array manipulations that return action of the driver,
                    # in terms of permutation indices.

    def new_mixerX(self):
        def split(x,k):
            return x.reshape((2**k,-1))
        def sym_swap(x):
            return np.asarray([x[-1],x[-2],x[1],x[0]])
        n = self.n
        x_list = []
        t1 = np.asarray([np.arange(2**(n-1),2**n),np.arange(0,2**(n-1))])
        t1 = t1.flatten()
        x_list.append(t1.flatten())
        t2 = t1.reshape(4,-1)
        t3 = sym_swap(t2)
        t1 = t3.flatten()
        x_list.append(t1)
        k = 1
        while k < (n-1):
            t2 = split(t1,k)
            t2 = np.asarray(t2)
            t1=[]
            for y in t2:
                t3 = y.reshape((4,-1))
                t4 = sym_swap(t3)                
                t1.append(t4.flatten())
            t1 = np.asarray(t1)
            t1 = t1.flatten()
            x_list.append(t1)
            k+=1
        return x_list
    #__________________________________________________________________________________________________________


    def U_gamma(self,angle,state):       # applies exp{-i\gamma H_z}, here as "U_gamma", on a "state".
        t = -1j*angle
        H_new = self.H.reshape(2**self.n,1)
        state = state*np.exp(t*self.H.reshape(2**self.n,1))
        return state

    
    def V_beta(self,angle,state):        # applies exp{-i\beta H_x}, here as "V_beta", on a "state".
        c = np.cos(angle)
        s = np.sin(angle)
        for i in range(self.n):
            t = self.X[i]
            st = state[t]
            state = c*state + (-1j*s*st)
        return state

    #__________________________________________________________________________________________________________

                        # This step creates the qaoa_ansatz w.r.t to "angles" that are passed.
                        # "angles" are passed as [gamma_1,gamma_2,...,gamma_p,beta_1,beta2,....beta_p].

    def qaoa_ansatz(self, angles):
        state = np.ones((2**self.n,1),dtype = 'complex128')*(1/np.sqrt(2**self.n))
        for i in range(self.p):
            state = self.U_gamma(angles[i],state)
            state = self.V_beta(angles[self.p + i],state)
        return state

    #__________________________________________________________________________________________________________


    def expectation(self,angles):   # Calculates expected value of the Hamiltonian w.r.t qaoa_ansatz state,
                                           # defined by the specific choice of "angles".                                                    
        state = self.qaoa_ansatz(angles)
        ex = np.vdot(state,state*(self.H).reshape((2**self.n,1)))
        return  np.real(ex)
    

    def overlap(self,state):        # Calculates ground state overlap for any "state",
                                                        # passed to it. Usually the final state or "f_state" returned,
                                                        # after optimization.                                          
        g_ener = min(self.H)
        olap = 0
        for i in range(len(self.H)):
            if self.H[i] == g_ener:
                olap+= np.absolute(state[i])**2
        return olap

   #__________________________________________________________________________________________________________

                    # Main execution of the algorithm.
                    # 1) Create "initial_angles", this would be the guess or
                    #    starting point for the optimizer.
                    # 2) Optimizer "L-BFGS-B" then takes "initial_angles" and calls "expectation".
                    # 3) "expectation" then returns a number and the optimizer tries to minimize this,
                    #     by doing finite differences. Thermination returns optimized angles,
                    #     stored here as "res.x".
                    # 4) Treating the optimized angles as being global minima for "expectation",
                    #    we calculate and store (as class attributes) the qaoa energy, here as "q_energy",
                    #    energy error, here as "q_error",
                    #    ground state overlap, here as "olap"
                    #    and also the optimal state, here as "f_state"

    def run(self):
        initial_angles=[]
        bds= [(0,2*np.pi+0.1)]*self.p + [(0,1*np.pi+0.1)]*self.p
        for i in range(2*self.p):
            if i < self.p:
                initial_angles.append(random.uniform(0,2*np.pi))
            else:
                initial_angles.append(random.uniform(0,np.pi))
        # start = time.perf_counter()
        res = minimize(self.expectation,initial_angles,method='L-BFGS-B', jac=None, bounds=bds, options={'maxiter': 1000})
        # elapsed = time.perf_counter() - start
        # res = differential_evolution(self.expectation, bds,maxiter=100, callback=None, disp=False, init='latinhypercube',workers=-1)
        self.q_energy = self.expectation(res.x)
        self.q_error = self.q_energy - self.min
        self.f_state = self.qaoa_ansatz(res.x)
        self.olap = self.overlap(self.f_state)[0]
        self.angles = res.x
        self.best_bitstring = np.binary_repr(int(np.argmax(np.abs(self.f_state))), width=self.n)
     #__________________________________________________________________________________________________________
