# -*- encoding: utf-8

# ToyQCMPS
# MPS program based on Ulrich Schollwöck, The density-matrix renormalization group in the age of matrix product states, Annals of Physics, 326 (2011), 96-192
import numpy as np
import math                        
print(" /$$$$$$$$                     /$$$$$$   /$$$$$$  /$$      /$$ /$$$$$$$   /$$$$$$ ")
print("|__  $$__/                    /$$__  $$ /$$__  $$| $$$    /$$$| $$__  $$ /$$__  $$")
print("   | $$  /$$$$$$  /$$   /$$| | $$  \ $$| $$  \__/| $$$$  /$$$$| $$  \ $$| $$  \__/")
print("   | $$ /$$__  $$| $$  | $$| | $$  | $$| $$      | $$ $$/$$ $$| $$$$$$$/|  $$$$$$ ")
print("   | $$| $$  \ $$| $$  | $$| | $$  | $$| $$      | $$  $$$| $$| $$____/  \____  $$")
print("   | $$| $$  | $$| $$  | $$| | $$/$$ $$| $$    $$| $$\  $ | $$| $$       /$$  \ $$")
print("   | $$|  $$$$$$/|  $$$$$$$| |  $$$$$$/|  $$$$$$/| $$ \/  | $$| $$      |  $$$$$$/")
print("   |__/ \______/  \____  $$   \____ $$$ \______/ |__/     |__/|__/       \______/ ")
print("                  /$$  | $$        \__/                                           ")
print("                 |  $$$$$$/                                                       ")
print("                  \______/                                                        ")

debug = False

class TensorError(Exception):
    pass

def tensorError(tensor,site:int, message='TensorError: Something wrong with tensor'):
    print(message, 'at site ',index)
    print(tensor.shape)
    if debug == True:
        print(tensor)
    raise TensorError("")

def fakeMPO(num_sites:int,phys_dim:int):
    MPO = []
    MPO.append(np.ones((1,10,phys_dim,phys_dim)))
    for i in range(1,num_sites-1):
        MPO.append(np.ones((10,10,phys_dim,phys_dim)))
    MPO.append(np.ones((10,1,phys_dim,phys_dim)))
    return MPO

class Site(object):

    #      --*-- 0                 0 --*--
    #        |                         |
    # (L)  --#-- 1       1         1 --#-- (R)
    #        |           |             |
    #      --*-- 2   0 --*-- 2     2 --*--
    #       i-1          i            i+1
    def __init__(self, bond_l, phys_dim, bond_r, index, neig = 1, bond_dimension_thresh=0, bond_error_thresh=0):
        self._site_tensor  = np.random.random((bond_l,phys_dim,bond_r))
        self.site_num     = index
        self.max_bond     = bond_dimension_thresh
        self.max_error    = bond_error_thresh
        self.nroots       = neig
        self.E            = None
        self.L            = None
        self.R            = None
        self.lock         = False

    @property
    def bond_l(self):
        return self.site_tensor.shape[0]

    @property
    def bond_r(self):
        return self.site_tensor.shape[2]

    @property
    def phys_dim(self):
        return self.site_tensor.shape[1]

    @property
    def site_tensor(self):
        return self._site_tensor

    @site_tensor.setter
    def site_tensor(self,new_state_tensor):
        if self.phys_dim != new_state_tensor.shape[1]:
            errMsg  = "Cannot set new tensor for this site: " + str(self.site_num)
            errMsg += "\nPhysical indices do not corresponds."
            raise Exception(errMsg)
        self._site_tensor = new_state_tensor
        self.lock = False

    def svd_decompose(self, direction):
        if direction == 'left':
            u,s,v = np.linalg.svd(self.site_tensor.reshape(self.bond_l*self.phys_dim,self.bond_r),
                        full_matrices=False)
        elif direction == 'right':
            u,s,v = np.linalg.svd(self.site_tensor.reshape(self.bond_l,self.phys_dim*self.bond_r),
                        full_matrices=False)
        else:
            raise("Wrong direction for SVD.")
        if self.max_bond > 0:
            return u[:,:self.max_bond], s[:self.max_bond], v[:self.max_bond,:]
        elif self.max_error > 0:
            sum = 0
            i = 0
            max_bond_dim = 0
            for i in s:
                sum += i/s.sum()
                max_bond_dim += 1
                if sum > 1-self.max_error:
                    break
            return u[:,:max_bond_dim],s[:max_bond_dim],v[:max_bond_dim,:]
        else:
            return u,s,v

    def left_canonicalize(self):
        if self.lock == True:
            raise RuntimeError("Site is locked. Trying to to left canonicalization on already canonicalized site")
        self.lock = True
        u,s,v = self.svd_decompose('left')
        self.site_tensor = u.reshape((self.bond_l,self.phys_dim,-1))
        return np.dot(np.diag(s),v)

    def right_canonicalize(self):
        if self.lock == True:
            raise RuntimeError("Site is locked. Trying to to right canonicalization on already canonicalized site")
        self.lock = True
        u,s,v = self.svd_decompose('right')
        self.site_tensor = v.reshape((-1,self.phys_dim,self.bond_r))
        return np.dot(u,np.diag(s))

    def calc_E(self, ket_site, mpo = None, first=None):
        if self.site_tensor.shape[1] != ket_site.site_tensor.shape[1]:
            raise ValueError("Physical indices doesn't align.")
        if mpo is not None:
        #  0 --*-- 2          2         0 --*-- 1                           0 --*-- 1
        #      |              |             |                                   |
        #      1      x   0 --#-- 1 --> 2 --#-- 3   x       1      -->      2 --#-- 3
        #                     |             |               |                   |
        #                     3             4           0 --*-- 2           4 --*-- 5
            top_contraction = np.tensordot(ket_site.site_tensor.conj(), mpo, axes=[1,2])
            self.E = np.tensordot(top_contraction, self.site_tensor, axes=[4,1])
            if first == 'right' :
                self.E = self.E.transpose(0,2,4,1,3,5)
                self.R = self.E = self.E.reshape(ket_site.site_tensor.shape[0],mpo.shape[0],self.site_tensor.shape[0])
            if first == 'left' :
                self.E = self.E.transpose(1,3,5,0,2,4)
                self.R = self.E = self.E.reshape(ket_site.site_tensor.shape[1],mpo.shape[1],self.site_tensor.shape[1])
        else:
        #  0 --*-- 2                           0 --*-- 1
        #      |                                   |
        #      1       x       1      -->          |
        #                      |                   |
        #                  0 --*-- 2           2 --*-- 3
            self.E = np.tensordot(ket_site.site_tensor.conj(),self.site_tensor, axes=[1,1])
            #print(self.E.shape, ket_site._site_tensor.conj().shape,self._site_tensor.shape)
        return self.E

    def calc_nextL(self, mpo):
        if self.L is None and self.site_num > 0:
            errMsg = "This site L intermediate is unavailable. Site is " + str(self.site_num)
            raise RuntimeError(errMsg)
        if mpo.shape[2] != mpo.shape[3] or mpo.shape[2] != self.site_tensor.shape[1]:
            raise ValueError("Wrong dimension of MPO and site tensor")
        #  *-- 0               0 --*-- 2        *-------*-- 3
        #  |                       |            |       |
        #  #-- 1           x       1      -->   #-- 0   2
        #  |                                    |
        #  *-- 2                                *-- 1
        top_contraction = np.tensordot(self.L, self.site_tensor.conj(), axes=[0,0])
        #  *-------*-- 3           2            *-------*-- 1
        #  |       |               |            |       |
        #  #-- 0   2       x   0 --#-- 1  -->   #-------#-- 2
        #  |                       |            |       |
        #  *-- 1                   3            *-- 0   3
        mid_contraction = np.tensordot(top_contraction,mpo, axes=[[0,2],[0,2]])
        #  *-------*-- 1                        *-- 0
        #  |       |                            |
        #  #-------#-- 2   x       1      -->   #-- 1
        #  |       |               |            |
        #  *-- 0   3           0 --*-- 2        *-- 2
        full_contraction = np.tensordot(mid_contraction, self.site_tensor, axes = [[0,3],[0,1]])
        nextL = full_contraction
        return nextL

    def calc_nextR(self, mpo):
        #if self.R is not None and self.site_num < 0:
        #    errMsg = "This site R intermediate is unavaible. Site is " + str(self.site_num)
        #    raise RuntimeError(errMsg)        
        if mpo.shape[2] != mpo.shape[3] or mpo.shape[2] != self.site_tensor.shape[1]:
            raise ValueError("Wrong dimension of MPO and site tensor")
        #  0 --*-- 2               0 --*        0 --*-------*
        #      |                       |            |       |
        #      1               x   1 --#  -->       1   2 --#
        #                              |                    |
        #                          2 --*                3 --*
        top_contraction = np.tensordot(self.site_tensor.conj(),self.R, axes=[2,0])
        #      2           0 --*-------*        2 --*-------*
        #      |               |       |            |       |
        #  0 --#-- 1   x       1   2 --#  -->   0 --#-------#
        #      |                       |            |       |
        #      3                   3 --*            1   3 --*
        mid_contraction = np.tensordot(mpo,top_contraction, axes=[[2,1],[0,1]])
        #                  2 --*-------*        2 --*           0 --*
        #                      |       |            |               |
        #      1       x   0 --#-------#  -->   1 --#    -->    1 --#
        #      |               |       |            |               |
        #  0 --*-- 2           1   3 --*        0 --*           2 --*
        full_contraction = np.tensordot(self.site_tensor,mid_contraction, axes = [[1,2],[1,3]])
        nextR = full_contraction.transpose(2,1,0)
        return nextR

    def clear_R(self):
        self.R = None
        return

    def clear_L(self):
        self.L = None
        return

    def variational_contraction(self, mpo):
        if mpo.shape[2] != mpo.shape[3] or mpo.shape[2] != self.site_tensor.shape[1]:
            raise ValueError("Wrong dimension of MPO and site tensor")
        #  *-- 0                   2            *-- 0   3
        #  |                       |            |       |
        #  #-- 1           x   0 --#-- 1  -->   #-------#-- 2
        #  |                       |            |       |
        #  *-- 3                   3            *-- 1   4
        L_mpo_contraction = np.tensordot(self.L, mpo, axes=[1,0])
        #  *-- 0   3           0 --*            *-- 0   2   4 --*
        #  |       |               |            |       |       |
        #  #-------#-- 2  x    1 --#      -->   #-------#-------#
        #  |       |               |            |       |       |
        #  *-- 1   4           2 --*            *-- 1   3    5--*
        full_contraction = np.tensordot(L_mpo_contraction, self.R, axes=[2,1])
        return full_contraction.reshape(self.bond_l*self.phys_dim*self.bond_r,self.bond_l*self.phys_dim*self.bond_r)

    def variational_update(self, mpo, direction, next_site):
        H = self.variational_contraction(mpo)
        e,A = self.davidson(H,self.site_tensor.reshape(self.bond_l*self.phys_dim*self.bond_r))
        self.site_tensor = A.reshape(self.bond_l,self.phys_dim,self.bond_r)

        if direction == 'right':
            next_site.site_tensor = np.tensordot(self.left_canonicalize(),next_site.site_tensor, axes = [1,1])
            next_site.L = self.calc_nextL(mpo)

        if direction == 'left':
            next_site.site_tensor = np.tensordot(next_site.site_tensor,self.right_canonicalize(), axes = [2,1])
            next_site.R = self.calc_nextR(mpo)

        return e

    def davidson(self,H,guess):

        iter = 0
        k = 8
        n = H.shape[0]
        mmax = n//2
        B = np.eye(n,k)
        B[:,guess.shape[1]] = guess
        V = np.zeros((n,k))
        I = np.eye(n)
        ritz_vector = np.zeros(n)
        delta_den = np.zeros((n,n))
        eig_old = 1000

        for i in range(k):
           V[:,i] = B[:,i]/np.linalg.norm(B[:,i])

        i = k

        while i < mmax:
            print('I = ',i,' m = ',V.shape[1])
            iter = iter + 1
            G = np.linalg.multi_dot([V.transpose(),H,V])
            e,v = np.linalg.eig(G)
            sort_idx = e.argsort()
            e_sorted = e[sort_idx]
            v_sorted = v[:,sort_idx]

            for ii in range(i):
                delta_den = np.diag(1.0/np.diag(np.diag(H)-e_sorted[ii]*I))
                ritz_vector = np.dot(delta_den,np.linalg.multi_dot([H-e_sorted[ii]*I,V[:,:i],v_sorted[:,ii]]))
                ritz_vector = ritz_vector/(np.linalg.norm(ritz_vector))
                V = np.concatenate((V,ritz_vector[:,None]),axis = 1)

            i = V.shape[1]

            #ortogonalize
            if np.linalg.norm(eig_old - e_sorted[:self.nroots]) < 1e-7:
                 for ii in range(self.nroots):
                     print(e_sorted[ii],"Final eigenvalue")
                 return e_sorted[:self.nroots],v_sorted[:,:self.nroots]
                 break
            else:
                 eig_old = e_sorted[:self.nroots]
                 q, r = np.linalg.qr(V)
                 V = q


    def transfer_matrix(self):
        return self.E

class MatrixProductState(object):

    def __init__(self, phys_dim, nsites, tol = 1e-6, max_bond = 0, init_bond = 10, nroots=1):
        self.phys_dim   = phys_dim
        self.nsites   = nsites
        self.tol        = tol
        self.sites      = []
        self.init_bond  = init_bond
        self.max_bond   = max_bond
        self.nroots     = nroots
        self.lock_mps   = False
        self.norm       = 1.0
        self._H          = None
    
    @property
    def H(self):
        return self._H
    @H.setter
    def H(self, H, override = False):
        if self.H is not None or override is True:
            self._H = H

    def printMPS(self,header=True,conj=False):
        first_line  = ""
        second_line = ""
        third_line  = ""

        for i in range(self.nsites):
            first_line  +="   "+str(self.sites[i].phys_dim)+"   "
            second_line +="   |   "
            third_line +=str(self.sites[i].bond_l)+"--*--"
        third_line += str(self.sites[-1].bond_r)

        if header is True:
            print("\nMPS graphical representation")

        if conj is True:            
            print(third_line)
            print(second_line)
            print(first_line)
        else:
            print(first_line)
            print(second_line)
            print(third_line)

    def fillMPS(self):
        for i in range(self.nsites):
            self.addsite()

    def addsite(self):
        if self.lock_mps == True:
            raise RuntimeError("MPS is finalized. Cannot add another site.")
        if len(self.sites) == 0:
            self.sites.append(Site(1,self.phys_dim,self.init_bond,len(self.sites)+1,self.nroots,self.max_bond,self.tol))
        elif len(self.sites) == self.nsites-1:
            self.sites.append(Site(self.init_bond,self.phys_dim,1,len(self.sites)+1,self.nroots,self.max_bond,self.tol))
            self.lock_mps = True
        else:
         #   if len(self.sites) != len(self.mpo):
         #       raise ValueError("There is a mismatch in the number of sites and mpo")
            self.sites.append(Site(self.init_bond, self.phys_dim, self.init_bond, len(self.sites)+1, self.nroots, self.max_bond, self.tol))

    def to_vector(self):
        psi = self.sites[0].site_tensor

        for i in range(1,len(self.sites)):
            A = self.sites[i]
            #     1           1             1  2 
            #     |     +     |     -->     |  |
            # 0 --*-- 2   0 --*-- 2     0 --*--*-- 3
            psi = np.tensordot(psi,self.sites[i].site_tensor, axes=[2,0])
            #     1  2                      1
            #     |  |          -->         |
            # 0 --*--*-- 3              0 --*-- 2
            
            psi = psi.reshape(psi.shape[0],psi.shape[1]*psi.shape[2],psi.shape[3])
        if psi.shape[0] != 1 and psi.shape[2] != 1:
                raise TensorError("Wrong dimensions")

        psi = psi.reshape(-1)
        
        return psi

    def orthonormalize(self,direction = 'left'):
        if len(self.sites) < 3:
            raise RuntimeError("Orthogonalizing  MPS with less than 3 sites")

        if direction == 'left':
            for isite in range(len(self.sites)-1):
                next_site = self.sites[isite+1]
                curr_site = self.sites[isite]
                next_site.site_tensor = np.tensordot(curr_site.left_canonicalize(), next_site.site_tensor, axes = [1,0])
            left_norm_tensor = self.sites[-1].left_canonicalize()

            if left_norm_tensor.shape != (1,1):
                raise TypeError("Left normalization is not scalar")
            self.left_normalization = left_norm_tensor[0][0]

            if self.left_normalization < 0:
                self.left_normalization = -1*self.left_normalization
                self.sites[-1] = -1*self.sites[-1]
            return self.left_normalization
            #self.norm = 1/np.sqrt(self.left_normalization)

        elif direction == 'right':
            for isite in reversed(range(1,len(self.sites))):
                prev_site = self.sites[isite-1]
                curr_site = self.sites[isite]
                prev_site.site_tensor = np.tensordot(next_site.site_tensor, curr_site.right_canonicalize(), axes = [2,0])
            right_norm_tensor = self.sites[1].right_canonicalize()

            if right_norm_tensor.shape != (1,1):
                raise TypeError("Right normalization is not scalar")
            self.right_normalization = right_norm_tensor[0][0]

            if self.right_normalization < 0:
                self.right_normalization = -1*self.right_normalization
                self.sites[1] = -1*self.sites[1]
            return self.right_normalization
            #self.norm = 1/np.sqrt(self.right_normalization)
        else:
            raise RuntimeError("Wrong direction for orthogonalization")

    def vdot(self, ketMPS):
        if len(self.sites) != len(ketMPS.sites):
            raise ValueError("Incompatible number of sites for two MPS")
  
        bracket = self.sites[0].calc_E(ketMPS.sites[0])

        for isite in range(1,len(self.sites)):
            this_site = self.sites[isite]
            ket_site = ketMPS.sites[isite]
            bracket = np.tensordot(bracket,this_site.calc_E(ket_site),axes = [[1,3],[0,2]]).transpose(0,2,1,3)
      
        assert(bracket.shape==(1,1,1,1))
        return bracket[0][0][0][0]*self.norm*ketMPS.norm

    def norm(self):
        return np.sqrt(self.vdot(self))

    def find_ground_state(self, options):
        if self.H is None:
            raise RuntimeError("No hamiltonian was found")
        e_left = e_right = 1000.0
        energy = [0 for _ in range(self.nsites)]

        for i in range(self.nsites):
            self.sites[i].left_canonicalize()

        self.sites[self.sites-1].calc_E(self.H,first='right')
        for i in reverse(range(2,self.nsites)):
            self.sites[i-1].R = self.sites[i].calc_nextR()

        for i in range(options.nsweeps):
            for i in range(self.nsites):
                energy[i] = self.sites[i].variational_update(self.H,'right',self.sites[i+1])
            e_left = energy[math.floor(i/2.0)]
            for i in reverse(range(self.nsites)):
                energy[i] = self.sites[i].variational_update(self.H,'left',self.sites[i-1])
            e_right = energy[math.floor(i/2.0)]
            if math.abs(e_left-e_right) < options.conv:
                break
        E_0 = e_right
        return E_0

class MPO(object):
    def __init__(self, numsites):
        

    #def __add__(self,mps2):

myMPO = fakeMPO(4,4)
for i in myMPO:
    print(i.shape)

myMPS = MatrixProductState(4,4,max_bond = 10)    
myMPS.fillMPS()
myMPS.printMPS(True,True)
myMPS.printMPS(False)
psi0 = myMPS.to_vector()
norm = myMPS.orthonormalize('left')
myMPS.printMPS(True,True)
myMPS.printMPS(False)
psi = myMPS.to_vector()
print("===========")
print(1-np.linalg.norm(psi))
print(np.linalg.norm(psi0-norm*psi))
print("===========")
print(myMPS.vdot(myMPS))


