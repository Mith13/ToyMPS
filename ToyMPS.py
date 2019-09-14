# -*- encoding: utf-8

# ToyQCMPS
# MPS program based on Ulrich Schollwöck, The density-matrix renormalization group in the age of matrix product states, Annals of Physics, 326 (2011), 96-192

print("ToyQCMPS")


class Site(object):
    def __init__(self, bond_l,phys_dim,bond_r, index, bond_dimension_thresh=None, bond_error_thresh=None):
        self._site_tensor = np.random.random((bond_l,phys_dim,bond_r))
        self.site_num     = index
        self.max_bond     = bond_dimension_thresh
        self.max_error    = bond_error_thresh
        #      --*-- 0                 0 --*--
        #        |                         |
        # (L)  --#-- 1       1         1 --#-- (R)
        #        |           |             |
        #      --*-- 2   0 --*-- 2     2 --*--
        #       i-1          i            i+1
        self.E            = None
        self.L            = None
        self.R            = None

    @property
    def bond_l(self):
        return self.site_tensor[0]

    @property
    def bond_r(self):
        return self.site_tensor[2]

    @property
    def phys_dim(self):
        return self.site_tensor[1]

    @property
    def site_tensor(self):
        return self._site_tensor

    @site_tensor.setter
    def site_tensor(self,new_state_tensor):
        if self.phys_dim != new_state_tensor[1]:
            errMsg  = "Cannot set new tensor for this site: " + str(self.site_num)
            errMsg += "\nPhysical indices do not corresponds."
            raise Exception(errMsg)
        self._site_tensor=new_state_tensor
        self.purge_intermediates()

    def clear_intermediates():
        return

    def decompose(self, direction):
        if direction == 'left':
            u,s,v = np.linalg.svd(self.site_site_tensor.reshape(self.bond_l*self.phys_dim,self.bond_r),
                        full_matrices=False)
        elif direction == 'right':
            u,s,v = np.linalg.svd(self.site_site_tensor.reshape(self.bond_l,self.phys_dim*self.bond_r),
                        full_matrices=False)
        else:
            raise("Wrong direction for SVD.")
        if self.max_bond is not None:
            return u[:,:self.max_bond], s[:self.max_bond], v[:self.max_bond,:]
        elif self.max_error is not None:
            sum = 0
            i = 0
            for i in s:
                sum += i/s.sum()
                i += 1
                if sum > 1-self.max_error:
                    break
            max_bond_dim = i
            return u[:,:max_bond_dim], s[:max_bond_dim], v[:max_bond_dim,:]
        else:
            return u,s,v

    def left_canonicalize(self):
        u,s,v = decompose('left')
        self.matrix = u.reshape((self.bond_l,self.phys_dim,-1))
        return np.dot(np.diag(s),v)

    def right_canonicalize(self):
        u,s,v = decompose('right')
        self.matrix = u.reshape((-1,self.phys_dim,self.bond_r))
        return np.dot(np.diag(s),v)

    def calc_E(self, site2):

        if self.mpo is not None:
        #  0 --*-- 2          2         0 --*-- 1                           0 --*-- 1
        #      |              |             |                                   |
        #      1      x   0 --#-- 1 --> 2 --#-- 3   x       1      -->      2 --#-- 3
        #                     |             |               |                   |
        #                     3             4           0 --*-- 2           4 --*-- 5
            top_contraction = np.tensordot(self.matrix.conj(), self.mpo, axes=[1,2])
            self.E = np.tensordot(top_contraction, self.matrix, axes=[4,1])
        else:
        #  0 --*-- 2                           0 --*-- 1
        #      |                                   |
        #      1       x       1      -->          |
        #                      |                   |
        #                  0 --*-- 2           2 --*-- 3
            if self.mpo.shape is not site2.mpo.shape:
               errMsg = "Two sites "+ str(self.site_num) + "," + str(site2.site_num) + " have different MPO"
               raise(errMsg)
            self.E = np.tensordot(self.matrix.conj(),self.matrix, axes=[1,1])
        return self.E

    def calc_nextL(self):
        if self.L is None and self.site_num > 0:
            errMsg = "This site L intermediate is unavailable. Site is " + str(self.site_num)
            raise(errMsg)
        #  *-- 0               0 --*-- 2        *-------*-- 3
        #  |                       |            |       |
        #  #-- 1           x       1      -->   #-- 0   2
        #  |                                    |
        #  *-- 2                                *-- 1
        top_contraction = np.tensordot(self.L, self.matrix.conj(), axes=[0,0])
        #  *-------*-- 3           2            *-------*-- 1
        #  |       |               |            |       |
        #  #-- 0   2       x   0 --#-- 1  -->   #-------#-- 2
        #  |                       |            |       |
        #  *-- 1                   3            *-- 0   3
        mid_contraction = np.tensordot(top_contraction,self.mpo, axes=[[0,2],[0,2]])
        #  *-------*-- 1                        *-- 0
        #  |       |                            |
        #  #-------#-- 2   x       1      -->   #-- 1
        #  |       |               |            |
        #  *-- 0   3           0 --*-- 2        *-- 2
        full_contraction = np.tensordot(mid_contraction, self.matrix, axes = [[0,3],[0,1]])
        return nextL

    def calc_nextR(self, previous_R):
        if self.R is not None and self.site_num > 0:
            errMsg = "This site R intermediate is unavaible. Site is " + str(self.site_num)
            raise(errMsg)
        #  0 --*-- 2               0 --*        0 --*-------*
        #      |                       |            |       |
        #      1               x   1 --#  -->       1   2 --#
        #                              |                    |
        #                          2 --*                3 --*
        top_contraction = np.tensordot(self.matrix.conj(),self.R, axes=[2,0])
        #      2           0 --*-------*        2 --*-------*
        #      |               |       |            |       |
        #  0 --#-- 1   x       1   2 --#  -->   0 --#-------#
        #      |                       |            |       |
        #      3                   3 --*            1   3 --*
        mid_contraction = np.tensordot(self.mpo,top_contraction, axes=[[2,1],[0,1]])
        #                  2 --*-------*        2 --*           0 --*
        #                      |       |            |               |
        #      1       x   0 --#-------#  -->   1 --#    -->    1 --#
        #      |               |       |            |               |
        #  0 --*-- 2           1   3 --*        0 --*           2 --*
        full_contraction = np.tensordot(self.matrix,mid_contraction, axes = [[1,2],[1,3]])
        nextr = full_contraction.transpose(2,1,0)
        return nextR

    def clear_R(self):
        self.R = None
        return

    def clear_L(self):
        self.L = None
        return

    def variational_contraction(self):
        #  *-- 0                   2            *-- 0   3
        #  |                       |            |       |
        #  #-- 1           x   0 --#-- 1  -->   #-------#-- 2
        #  |                       |            |       |
        #  *-- 3                   3            *-- 1   4
        L_mpo_contraction = np.tensordot(L_mps_contraction, self.mpo, axes=[1,0])
        #  *-- 0   3            0 --*            *-- 0   2   4 --*
        #  |       |                |            |       |       |
        #  #-------#-- 2   x    1 --#     -->    #-------#-------#
        #  |       |                |            |       |       |
        #  *-- 1   4            2 --*            *-- 1   3    5--*
        full_contraction = np.tensordot(L_mpo_contraction, self.R, axes=[2,1])
        return full_contraction.reshape(self.bond_l*self.phys_dim*self.bond_r,self.bond_l*self.phys_dim*self.bond_r)

    def variational_update(self, direction, previous_site, next_site):
        calc_nextL(previous_site.L)
        calc_nextR(next_site.R)
        H = variational_contraction()
        e,A = davidson(H)
        self.matrix = A.reshape(self.bond_l,self.phys_dim,self.bond_r)


    def davidson(self,H):
        iter = 0
        n = H.shape[0]
        B = np.eye(n,k)
        V = np.zeros((n,k))
        I = np.eye(n)
        ritz_vector = np.zeros(n)
        delta_denom = np.zeros((n,n))
        eig_old = 1000
        neig = 1
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
                delta_denom = np.diag(1.0/np.diag(np.diag(H)-e_sorted[ii]*I))
                ritz_vector = np.dot(f,np.linalg.multi_dot([H-e_sorted[ii]*I,V[:,:i],v_sorted[:,ii]]))
                ritz_vector = ritz_vector/(np.linalg.norm(ritz_vector))
                V = np.concatenate((V,ritz_vector[:,None]),axis = 1)
            i = V.shape[1]

            #ortogonalize
            if np.linalg.norm(eig_old - e_sorted[:neig]) < 1e-7:
                 for ii in range(neig):
                     print(e_sorted[ii],"Final eigenvalue")
                 return e_sorted[:neig],v_sorted[:,:neig]
                 break
            else:
                 eig_old = e_sorted[:neig]
                 q, r = np.linalg.qr(V)
                 V = q


    def transfer_matrix(self):
        return self.E

class MatrixProductState(object):

    def __init(self):
        self._tensor=None



