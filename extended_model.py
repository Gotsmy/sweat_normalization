import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from multiprocessing import Pool
import tqdm

# changes with v2:
# added extended_model_numpy_pqn class
# made optimizable

def bateman(time,p):
    ''' 
    Calculates Bateman Concentration Time Series.
    -
    Input
    time      nd.array of shape (n,t) with n metabolites and t timepoints.
    p         nd.array of shape (5,n,t) with 5 kinetic parameters (ka, ke, c0, lag, d), n metabolites, and t timepoints
    - 
    Output
    y         nd.array of shape (n,t) of metabolite concentrations.
    '''
    with np.errstate(all='ignore'):
        y = p[0]/(p[1]-p[0])*(np.exp(-p[0]*(time-p[3]))-np.exp(-p[1]*(time-p[3])))*p[2]
        y = np.nan_to_num(y)
    return np.clip(y,a_min=0,a_max=np.inf)+p[4]

def fun_1(time,p):
    '''
    Simplified function describing a Bateman-like concentration time series.
    -
    Input
    time      nd.array of shape (n,t) with n metabolites and t timepoints.
    p         nd.array of shape (4,n,t) with 5 kinetic parameters (ke, c0, lag, d), n metabolites, and t timepoints
    - 
    Output
    y         nd.array of shape (n,t) of metabolite concentrations.
    '''
    y = p[1]*(time-p[2])*np.exp(-p[0]*time)
    y = np.clip(y,a_min=0,a_max=np.inf)+p[3]
    return y

# EM model
class extended_model_numpy():
    '''
    Builds a kinetic model for an arbitrary number of metabolites with the kinetic 
    function defined in self._func.
    '''
    
    def __init__(self,time,length,fun='bateman'):
        self.time   = time
        self.length = length
        self.size   = len(self.time)
        self._time_tensor = np.tile(self.time,self.length).reshape(self.length,-1)
        self.fun_name = fun
        self.sigma = np.ones(self.length*self.size)
        self._has_bounds = False
        self._has_metabolite_names = False
        self._has_measured_data = False
        self._has_pqn_data = False
        # optimization parameters
        self._is_optimized  = False
        self.SSE            = np.nan
        self._loss_function = None
        
        if fun == 'bateman':
            self._func  = bateman
            self._func_parameter_number = 5
            self.get_tensor_parameters = self.get_tensor_parameters_5
        elif fun == 'fun_1':
            self._func = fun_1
            self._func_parameter_number = 4
            self.get_tensor_parameters = self.get_tensor_parameters_4
        else:
            print('self._func could not be determined.')
            
        # generate initial parameters depending on how many are needed.
        self.parameters = np.concatenate((np.zeros(self.length*self._func_parameter_number),np.ones(len(self.time))),axis=0)
        
            
    def get_kinetic_parameters(self):
        return self.parameters[:self.length*self._func_parameter_number]
    
    def get_sweat_volumes(self):
        return self.parameters[self.length*self._func_parameter_number:]
    
    def update_parameters(self,parameters):
        assert len(parameters) == self.length*self._func_parameter_number+self.size, 'Shape of parameters is incorrect. {} should be {}..'.format(len(parameters),self.length*self._func_parameter_number+len(self.time))
        self.parameters = parameters
        
    def get_tensor_parameters_5(self):
        tmp = self.get_kinetic_parameters()
        p_k = np.repeat([tmp[0::5],tmp[1::5],tmp[2::5],tmp[3::5],tmp[4::5]],self.size).reshape(-1,self.length,self.size)
        return p_k

    def get_tensor_parameters_4(self):
        tmp = self.get_kinetic_parameters()
        p_k = np.repeat([tmp[0::4],tmp[1::4],tmp[2::4],tmp[3::4]],self.size).reshape(-1,self.length,self.size)
        return p_k
    
    def update_loss_function(self,loss_name):
        '''
        Define a Loss Function to use during self._optimize().
        -
        Input
        loss_name   Str. 
                    Default: None            (not implemented)
                    Options: 
                             Custom:    max_linear_loss, max_cauchy_loss
                             Built_in:  linear, huber, soft_l1, cauchy, arctan
        '''
        if loss_name == 'max_linear_loss':
            self._loss_function = self.max_linear_loss
        elif loss_name == 'max_cauchy_loss':
            self._loss_function = self.max_cauchy_loss
        elif loss_name in ['linear', 'huber', 'soft_l1', 'cauchy', 'arctan']:
            self._loss_function = loss_name
        else:
            print(loss_name,'not found. Loss NOT updated.')
        
    
    def fit(self,time,*parameters):
        '''
        Returns flattened M from the Equation M = C * V_sweat.
        self.parameters IS     updated.
        self.time       IS NOT updated.
        '''
        self.parameters = np.array(parameters)
        sweat_volumes = np.tile(self.get_sweat_volumes(),self.length).reshape(self.length,self.size)
        y = self._func(self._time_tensor,self.get_tensor_parameters())*sweat_volumes
        return y.flatten('F')
    
    def fit_tensor(self,time,*parameters):
        '''
        Returns unflattened M from the Equation M = C * V_sweat.
        self.parameters IS     updated.
        self.time       IS NOT updated.
        '''
        self.parameters = np.array(parameters)
        sweat_volumes = np.tile(self.get_sweat_volumes(),self.length).reshape(self.length,self.size)
        y = self._func(self._time_tensor,self.get_tensor_parameters())*sweat_volumes
        return y
    
    def plot(self,time,*parameters):
        '''
        Returns flattened C from the Equation M = C * V_sweat.
        self.parameters IS NOT updated.
        self.time       IS NOT updated.
        '''
        time_tensor = np.tile(time,self.length).reshape(self.length,-1)
        tmp_size = self.size
        tmp_parameters = self.parameters
        self.size=len(time)
        self.parameters = np.array(parameters)
        y = self._func(time_tensor,self.get_tensor_parameters())
        self.size=tmp_size
        self.parameters = tmp_parameters
        return y.flatten('F')
    
    def plot_tensor(self,time,*parameters):
        '''
        Returns unflattened C from the Equation M = C * V_sweat.
        self.parameters IS NOT updated.
        self.time       IS NOT updated.
        '''
        time_tensor = np.tile(time,self.length).reshape(self.length,-1)
        tmp_size = self.size
        tmp_parameters = self.parameters
        self.size=len(time)
        self.parameters = np.array(parameters)
        y = self._func(time_tensor,self.get_tensor_parameters())
        self.size=tmp_size
        self.parameters = tmp_parameters
        return y
    
    def update_metabolite_names(self,metabolite_names):
        '''
        If metabolites are named it is possible to save their names in the model class.
        -
        Input
        metabolite_names    List or nd.array of shape (self.length).
        '''
        assert len(metabolite_names) == self.length
        self.metabolite_names = metabolite_names
        self._has_metabolite_names = True
        
    def update_fit_bounds(self,lower_bounds,upper_bounds):
        '''
        Set bounds for model optimization. They are parsed into the scipy.optmize.curve_fit function.
        Lower bounds have to be always lower than upper bounds.
        -
        Input
        lower_bounds    List or array of shape (len(self.parameters)).
        upper_bounds    List or array of shape (len(self.parameters)).
        '''
        assert len(upper_bounds) == len(self.parameters)
        self.upper_bounds = np.array(upper_bounds)
        assert len(lower_bounds) == len(self.parameters)
        self.lower_bounds = np.array(lower_bounds)
        self._has_bounds = True
        
    def update_sigma(self,sigma):
        '''
        Set sigma for model optimization. It is parsed into the scipy.optimize.curve_fit function.
        There the weighted error residuals are calculated according to chisq = sum((r / sigma) ** 2).
        - 
        Input
        sigma    List or array of shape (self.size * self.length).
        '''
        assert len(sigma) == self.size*self.length
        self.sigma = np.array(sigma)
        
    def update_measured_data(self,measured_data):
        '''
        Set measured data for model optimization (equal to M in M = C * V_sweat). It is parsed into the 
        scipy.optimize.curve_fit function and the model will be optimized to be as close as possible to it. 
        -
        Input
        measured_data    Flattened array of measured data of shape (self.size * self.length).
        '''
        assert len(measured_data) == self.length*self.size
        self.measured_data = measured_data
        self._has_measured_data = True
        
    def info(self):
        '''
        Convenience function that returns a pd.DataFrame with some overview information about self.
        -
        Output
        DataFrame    pd.DataFrame
        '''
        model_properties = {
            'length':[self.length],
            'size':[self.size],
            'fun':[self.fun_name],
            'parameters':[len(self.parameters)],
            'bounds':[self._has_bounds],
            'measured data':[self._has_measured_data],
            'metabolite names':[self._has_metabolite_names],
            'is optimized':[self._is_optimized],
            'optimization SSE':[self.SSE]
        }
        return pd.DataFrame(model_properties,index=['properties']).transpose()
    
    def _optimize(self,seed):
        '''
        Internal function that optimizes the model to given measured data.
        Measured data and bounds have to be parsed with self.update_measured_data() and 
        self.update_bounds().
        -
        Input
        seed    Seed variable for np.random.seed(seed)
        -
        Output
        array   Numpy array of shape (len(self.paramters)+1) of parameters + SSE of fitted model.
        '''
        
        assert self._has_measured_data, 'No measured data parsed.'
        assert self._has_bounds, 'No bounds parsed.'
        np.random.seed(seed)
        p0 = np.random.uniform(self.lower_bounds,self.upper_bounds)
        try:
            parameters, variance = curve_fit(f       = self.fit,
                                             xdata   = self.time,
                                             ydata   = self.measured_data,
                                             sigma   = self.sigma,
                                             p0      = p0,
                                             bounds  = (self.lower_bounds,self.upper_bounds),
                                             method  = 'trf',
                                             max_nfev= 1000,
                                             loss    = self._loss_function,
                                             tr_solver = 'exact',
                                            )
            SSE = np.sum((self.measured_data-self.fit(self.time,*parameters))**2)
            return np.concatenate([parameters,[SSE]])
        except (RuntimeError) as e:
            return np.concatenate([self.lower_bounds*np.nan,[np.inf]])
            
    
    def optimize_monte_carlo(self,n_replicates,n_cpu):
        '''
        Function that optimizes the model to given measured data.
        Measured data and bounds have to be parsed with self.update_measured_data() 
        and self.update_bounds().
        -
        Input
        n_replicates    Int. Number of monte carlo replicates to fit.
        n_cpu           Int. Number of CPUs to use for multiprocessing.
        -
        Output
        array           Numpy array of shape (len(self.paramters)+1,n_replicates) of parameters
                        + SSE of fitted model times the number of replicates.

        '''
        
        assert self._has_measured_data, 'No measured data parsed.'
        assert self._has_bounds, 'No bounds parsed.'
        n_cpu = np.min([n_replicates,n_cpu])
        _input = list(range(n_replicates))
        
        # multiprocessing 
        _output = []
        with Pool(processes = n_cpu) as p:
            for _ in tqdm.tqdm(p.imap_unordered(self._optimize,_input),total=n_replicates):
                _output.append(_)
        
        _output = np.array(_output)
        r = len(self.parameters)+1
        best_parameter = _output[np.argmin(_output[:,-1]),-r:-1]
        self.update_parameters(best_parameter)
        self._is_optimized = True
        self.SSE = np.min(_output[:,-1])
        if type(self) == extended_model_numpy_pqn:
            self.x = _output[np.argmin(_output[:,-1]),0]
        return _output
    
    def max_linear_loss(self,absolute_error):
        '''
        Takes array of absolute error, calculates relative error. From the maximum of both Linear loss as implemented in SciPy is calculated.
        '''
        y = self.fit(self.time,*self.parameters)
        relative_error = np.divide(absolute_error, y, out=absolute_error.copy(), where=y!=0)
        # maximum error
        z = np.maximum(absolute_error,relative_error)
        rho = np.zeros((3,len(z)))
        rho[0] = z
        rho[1] = np.ones(len(z))
        return rho
    
    def max_cauchy_loss(self,absolute_error):
        '''
        Takes array of absolute error, calculates relative error. From the maximum of both Cauchy loss as implemented in SciPy is calculated.
        '''
        y = self.fit(self.time,*self.parameters)
        relative_error = np.divide(absolute_error, y, out=absolute_error.copy(), where=y!=0)
        # maximum error
        z = np.maximum(absolute_error,relative_error)
        rho = np.empty((3,len(z)))
        rho[0] = np.log1p(z)
        t = 1 + z
        rho[1] = 1 / t
        rho[2] = -1 / t**2
        return rho
    
# MIX model
class extended_model_numpy_pqn(extended_model_numpy):
    def __init__(self,time,length,fun='bateman'):
        super().__init__(time,length,fun='bateman')
        self.sigma = np.ones((self.length+1)*self.size)
        self.x = np.nan
        
        
    def update_fit_bounds(self,lower_bounds,upper_bounds):
        '''
        Set bounds for model optimization. They are parsed into the scipy.optmize.curve_fit function.
        Lower bounds have to be always lower than upper bounds.
        -
        Input
        lower_bounds    List or array of shape (len(self.parameters) + 1).
        upper_bounds    List or array of shape (len(self.parameters) + 1).
        '''
        assert len(upper_bounds) == len(self.parameters)+1
        self.upper_bounds = np.array(upper_bounds)
        assert len(lower_bounds) == len(self.parameters)+1
        self.lower_bounds = np.array(lower_bounds)
        self._has_bounds = True
        
    def update_measured_data(self,measured_data,pqn_data):
        '''
        Set measured data for model optimization (equal to M in M = C * V_sweat). It is parsed into the 
        scipy.optimize.curve_fit function and the model will be optimized to be as close as possible to it. 
        -
        Input
        measured_data    Flattened array of measured data of shape (self.size * self.length).
        '''
        assert len(np.concatenate([measured_data,pqn_data])) == (self.length+1)*self.size
        self.measured_data = np.concatenate([measured_data,pqn_data])
        self._has_measured_data = True
        
    def update_sigma(self,sigma):
        '''
        Set sigma for model optimization. It is parsed into the scipy.optimize.curve_fit function.
        There the weighted error residuals are calculated according to chisq = sum((r / sigma) ** 2).
        - 
        Input
        sigma    List or array of shape (self.size * self.length + 1).
        '''
        assert len(sigma) == self.size*(self.length+1)
        self.sigma = sigma
        
    def fit(self,time,x,*parameters):
        '''
        Returns flattened M from the Equation M = C * V_sweat concatenated to the sweat volume array.
        self.parameters IS     updated.
        self.x          IS     updated.
        self.time       IS NOT updated.
        '''
        self.parameters = np.array(parameters)
        self.x = x
        sweat_volumes = self.get_sweat_volumes()
        sweat_volumes_tensor = np.tile(sweat_volumes,self.length).reshape(self.length,self.size)
        y1 = self._func(self._time_tensor,self.get_tensor_parameters())*sweat_volumes_tensor
        y  = np.concatenate([y1.flatten('F'),sweat_volumes*x])
        return y
    
    def plot(self,time,x,*parameters):
        '''
        Returns flattened C from the Equation M = C * V_sweat concatenated to the sweat volume array.
        self.parameters IS NOT updated.
        self.time       IS NOT updated.
        '''
        assert len(parameters) == len(self.parameters)
        time_tensor = np.tile(time,self.length).reshape(self.length,-1)
        tmp_size = self.size
        tmp_parameters = self.parameters
        self.size=len(time)
        self.parameters = np.array(parameters)
        y1 = self._func(time_tensor,self.get_tensor_parameters())
        y2 = self.get_sweat_volumes()
        y  = np.concatenate([y1.flatten('F'),y2*x])
        self.size=tmp_size
        self.parameters = tmp_parameters
        return y
    
    def max_linear_loss(self,absolute_error):
        '''
        Takes array of absolute error, calculates relative error. From the maximum of both Linear loss as implemented in SciPy is calculated.
        '''
        y = self.fit(self.time,self.x,*self.parameters)
        relative_error = np.divide(absolute_error, y, out=absolute_error.copy(), where=y!=0)
        # maximum error
        z = np.maximum(absolute_error,relative_error)
        rho = np.zeros((3,len(z)))
        rho[0] = z
        rho[1] = np.ones(len(z))
        return rho
    
    def max_cauchy_loss(self,absolute_error):
        '''
        Takes array of absolute error, calculates relative error. From the maximum of both Cauchy loss as implemented in SciPy is calculated.
        '''
        y = self.fit(self.time,self.x,*self.parameters)
        relative_error = np.divide(absolute_error, y, out=absolute_error.copy(), where=y!=0)
        # maximum error
        z = np.maximum(absolute_error,relative_error)
        rho = np.empty((3,len(z)))
        rho[0] = np.log1p(z)
        t = 1 + z
        rho[1] = 1 / t
        rho[2] = -1 / t**2
        return rho