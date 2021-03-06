import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from multiprocessing import Pool
import tqdm

def bateman(time,p):
    ''' 
    Calculates a Modified Bateman Concentration Time Series.
    -
    Input
    time      numpy.ndarray of shape (n,t) with n metabolites and t timepoints.
    p         numpy.ndarray of shape (5,n,t) with 5 kinetic parameters (ka, ke, c0, lag, d), n metabolites, and t timepoints
    - 
    Output
    y         numpy.ndarray of shape (n,t) of metabolite concentrations.
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
    time      numpy.ndarray of shape (n,t) with n metabolites and t timepoints.
    p         numpy.ndarray of shape (4,n,t) with 4 kinetic parameters (ke, c0, lag, d), n metabolites, and t timepoints
    - 
    Output
    y         numpy.ndarray of shape (n,t) of metabolite concentrations.
    '''
    y = p[1]*(time-p[2])*np.exp(-p[0]*time)
    y = np.clip(y,a_min=0,a_max=np.inf)+p[3]
    return y

def standard_scale(array):
    '''
    Standard scales (Z-transforms) an array. I.e. scales an array to a mean of 0 and a standard deviation of 1.
    -
    Input
    array    numpy.ndarray.
    -
    Output
    z        Scaled numpy.ndarray.
    '''
    
    z = (array-np.mean(array))/np.std(array)
    return z

def mean_scale(array):
    '''
    Mean scales an array. I.e. scales an array to a mean of 1 by division.
    -
    Input
    array    numpy.ndarray.
    -
    Output
    m        Scaled numpy.ndarray.
    '''
    
    m = array/np.mean(array)
    return m

def log_mean_scale(array):
    '''
    Mean scales an array. I.e. scales an array to a mean of 0 by substraction.
    -
    Input
    array    numpy.ndarray.
    -
    Output
    m        Scaled numpy.ndarray.
    '''
    
    m = array-np.mean(array)
    return m

def no_scale(array):
    '''
    Returns the same array.
    '''
    return array

def log10px(array,x=1e-8):
    '''
    Calculates log10 of (array+x).
    -
    Input
    array    numpy.ndarray.
    x        float. default = 1e-8.
    '''
    return np.log10(array+x)

# EM model
class PKM_model():
    '''
    Builds a kinetic model for an arbitrary number of metabolites with the kinetic 
    function defined in self._fun.
    '''
    
    def __init__(self,time,n_metabolites,pkm_fun,trans_fun):
        '''
        Initialization.
        -
        Input
        time             numpy.ndarray of time points of measurements.
        n_metabolites    int. number of metabolites measured.
        pkm_fun          'bateman' or 'fun_1'. Function type used for PKM. 'fun_1' is a special case of 'bateman', where ka = ke.
        trans_fun        'log10' or 'none'. Function that transforms measured data.
        -
        Output
        PKM_model class
        '''
        self.time          = time
        self.n_metabolites = n_metabolites
        self.n_timepoints  = len(self.time)
        self._time_tensor  = np.tile(self.time,self.n_metabolites).reshape(self.n_metabolites,-1)
        self.pkm_fun_name  = pkm_fun
        self.trans_fun_name= trans_fun
        self.sigma         = np.ones(self.n_metabolites*self.n_timepoints)
        self._has_bounds   = False
        self._has_metabolite_names = False
        self._has_measured_data = False
        self._has_pqn_data = False
        # optimization parameters
        self._is_optimized = False
        self.loss           = np.nan
        self._loss_function= None
        
        if pkm_fun == 'bateman':
            self._fun  = bateman
            self._fun_parameter_number = 5
            self._get_tensor_parameters = self._get_tensor_parameters_5
        elif pkm_fun == 'fun_1':
            self._fun = fun_1
            self._fun_parameter_number = 4
            self._get_tensor_parameters = self._get_tensor_parameters_4
        else:
            raise ValueError('pkm_fun has to be either "bateman" or "fun_1".')
            
        if trans_fun == 'log10':
            self.trans_fun = log10px
        elif trans_fun == 'none':
            self.trans_fun = no_scale
        else:
            raise ValueError('trans_fun has to be either "log10" or "none".')
            
        # generate initial parameters depending on how many are needed.
        self.parameters = np.concatenate((np.zeros(self.n_metabolites*self._fun_parameter_number),np.ones(len(self.time))),axis=0)
        
    def set_metabolite_names(self,metabolite_names):
        '''
        If metabolites are named it is possible to save their names in the model class.
        -
        Input
        metabolite_names    List or numpy.ndarray of shape (self.n_metabolites).
        '''
        assert len(metabolite_names) == self.n_metabolites, 'Shape of metabolite_names is incorrect ({} should be {}).'.format(len(metabolite_names),self.n_metabolites)
        self.metabolite_names = metabolite_names
        self._has_metabolite_names = True
        
    def set_fit_bounds(self,lower_bounds,upper_bounds):
        '''
        Set bounds for model optimization. They are parsed into the scipy.optmize.curve_fit function.
        Lower bounds have to be always lower than upper bounds.
        -
        Input
        lower_bounds    List or array of shape (len(self.parameters)).
        upper_bounds    List or array of shape (len(self.parameters)).
        '''
        assert len(upper_bounds) == len(self.parameters), 'Shape of upper_bounds is incorrect ({} should be {}).'.format(len(upper_bounds),len(self.parameters))
        self.upper_bounds = np.array(upper_bounds)
        assert len(lower_bounds) == len(self.parameters), 'Shape of lower_bounds is incorrect ({} should be {}).'.format(len(lower_bounds),len(self.parameters))
        self.lower_bounds = np.array(lower_bounds)
        self._has_bounds = True
        
    def set_sigma(self,sigma):
        '''
        Set sigma for model optimization. It is parsed into the scipy.optimize.curve_fit function.
        There the weighted error residuals are calculated according to chisq = sum((r / sigma) ** 2). I. e. sigma is the reciprocal of lambda.
        - 
        Input
        sigma    List or array of shape (self.n_timepoints * self.n_metabolites).
        '''
        assert len(sigma) == self.n_timepoints*self.n_metabolites, 'Shape of sigma is incorrect ({} should be {}).'.format(len(sigma),self.n_timepoints*self.n_metabolites)
        self.sigma = np.array(sigma)
        
    def set_measured_data(self,measured_data):
        '''
        Set measured data for model optimization (equal to M in M = C * V_sweat). It is parsed into the 
        scipy.optimize.curve_fit function and the model will be optimized to be as close as possible to it. 
        -
        Input
        measured_data    Flattened array of measured data of shape (self.n_timepoints * self.n_metabolites).
        '''
        assert len(measured_data) == self.n_metabolites*self.n_timepoints, 'Shape of measured_data is incorrect ({} should be {}).'.format(len(measured_data),self.n_metabolites*self.n_timepoints)
        self.measured_data = measured_data
        self._has_measured_data = True

    def set_parameters(self,parameters):
        '''Updates the parameter values of the model.'''
        assert len(parameters) == self.n_metabolites*self._fun_parameter_number+self.n_timepoints, 'Shape of parameters is incorrect ({} should be {}).'.format(len(parameters),self.n_metabolites*self._fun_parameter_number+len(self.time))
        self.parameters = parameters

    def set_loss_function(self,loss_name):
        '''
        Define a Loss Function to use during self._optimize(). 
        Attention:
        Built_in loss functions don't work with the Monte Carlo approach used in self.optimize_monte_carlo().
        -
        Input
        loss_name   Str. 
                    Default: None            (not implemented)
                    Options: 
                             Custom:    max_linear_loss, max_cauchy_loss, cauchy_loss
                             Built_in:  linear, huber, soft_l1, cauchy, arctan
        '''
        if loss_name == 'max_linear_loss':
            self._loss_function = self.max_linear_loss
        elif loss_name == 'max_cauchy_loss':
            self._loss_function = self.max_cauchy_loss
        elif loss_name == 'cauchy_loss':
            self._loss_function = self.cauchy_loss
        elif loss_name in ['linear', 'huber', 'soft_l1', 'cauchy', 'arctan']:
            print('''Attention:
        Built_in loss functions don't work with the Monte Carlo approach used in self.optimize_monte_carlo().''')
            self._loss_function = loss_name
        else:
            raise ValueError('Loss not found.')
            
    def get_kinetic_parameters(self):
        '''Returns array of all kinetic parameters of the model.'''
        return self.parameters[:self.n_metabolites*self._fun_parameter_number]
    
    def get_sweat_volumes(self):
        '''Returns array of all sweat volume parameters of the model.'''
        return self.parameters[self.n_metabolites*self._fun_parameter_number:]
            
    def _get_tensor_parameters_5(self):
        '''Duplicates and reshapes kinetic parameters to be in a tensor of shape (5, self.n_metabolites, self.n_timepoints)'''
        tmp = self.get_kinetic_parameters()
        p_k = np.repeat([tmp[0::5],tmp[1::5],tmp[2::5],tmp[3::5],tmp[4::5]],self.n_timepoints).reshape(-1,self.n_metabolites,self.n_timepoints)
        return p_k

    def _get_tensor_parameters_4(self):
        '''Duplicates and reshapes kinetic parameters to be in a tensor of shape (4, self.n_metabolites, self.n_timepoints)'''
        tmp = self.get_kinetic_parameters()
        p_k = np.repeat([tmp[0::4],tmp[1::4],tmp[2::4],tmp[3::4]],self.n_timepoints).reshape(-1,self.n_metabolites,self.n_timepoints)
        return p_k
    
    def fit(self,time,*parameters):
        '''
        Returns flattened M from the Equation M = C * V_sweat.
        self.parameters IS     updated.
        self.time       IS NOT updated.
        -
        Input
        time           numpy.ndarray of time points for which M is calculated.
        *parameters    Parameters as floats. Lists or numpy.ndarrays lead to errors down the line.
        -
        Output
        y              numpy.ndarray of calculated M values of shape (self.n_metabolites*self.n_timepoints).
        '''
        self.parameters = np.array(parameters)
        sweat_volumes = np.tile(self.get_sweat_volumes(),self.n_metabolites).reshape(self.n_metabolites,self.n_timepoints)
        y = self._fun(self._time_tensor,self._get_tensor_parameters())*sweat_volumes
        y = self.trans_fun(y.flatten('F'))
        return y
        
    def plot(self,time,*parameters):
        '''
        Returns flattened C from the Equation M = C * V_sweat.
        self.parameters IS NOT updated.
        self.time       IS NOT updated.
        -
        Input
        time           numpy.ndarray of time points for which M is calculated.
        *parameters    Parameters as floats. Lists or numpy.ndarrays lead to errors down the line.
        -
        Output
        y              numpy.ndarray of calculated C values  of shape (self.n_metabolites*self.n_timepoints).
        '''
        time_tensor = np.tile(time,self.n_metabolites).reshape(self.n_metabolites,-1)
        tmp_n_timepoints = self.n_timepoints
        tmp_parameters = self.parameters
        self.n_timepoints=len(time)
        self.parameters = np.array(parameters)
        y = self._fun(time_tensor,self._get_tensor_parameters())
        self.n_timepoints=tmp_n_timepoints
        self.parameters = tmp_parameters
        return y.flatten('F')
    
    def get_M_tensor(self):
        '''
        Returns unflattened M from the Equation M = C * V_sweat.
        self.parameters IS     updated.
        self.time       IS NOT updated.
        -
        Input
        time           numpy.ndarray of time points for which M is calculated.
        *parameters    Parameters as floats. Lists or numpy.ndarrays lead to errors down the line.
        -
        Output
        y              numpy.ndarray of calculated M values  of shape (self.n_metabolites, self.n_timepoints).
        '''
        sweat_volumes = np.tile(self.get_sweat_volumes(),self.n_metabolites).reshape(self.n_metabolites,self.n_timepoints)
        y = self.trans_fun(self._fun(self._time_tensor,self._get_tensor_parameters())*sweat_volumes)
        return y
        
    def get_M_df(self):
        '''
        Returns DataFrame with unflattened, transformed M from the Equation M = C * V_sweat.
        self.parameters IS NOT updated.
        self.time       IS NOT updated.
        -
        Input
        None
        -
        Output
        y              pandas.DataFrame of shape (self.n_metabolites + 1, self.n_timepoints).
        '''
        assert self._has_metabolite_names, 'Model metabolite names are not set (see self.set_metabolite_names).'
        M_tensor = self.get_M_tensor()
        M_df = pd.DataFrame(index=np.arange(self.n_timepoints),dtype=float,columns=['time']+list(self.metabolite_names))
        M_df.loc[:,'time'] = self.time
        for i, metabolite in enumerate(self.metabolite_names):
            M_df.loc[:,metabolite] = M_tensor[i]
        return M_df

    
    def get_C_tensor(self,time=None,parameters=None):
        '''
        Returns unflattened C from the Equation M = C * V_sweat.
        self.parameters IS NOT updated.
        self.time       IS NOT updated.
        -
        Input
        time           None or numpy.ndarray of time points for which M is calculated. If None self.time is used.
        parameters     None or parameters as floats. Lists or numpy.ndarrays lead to errors down the line. If None self.parameters are used.
        -
        Output
        y              numpy.ndarray of calculated C values and sweat volumes of shape (self.n_metabolites, self.n_timepoints).
                       Sweat volumes are NOT returned!
        '''
        if type(time) == type(None):
            time = self.time
        if type(parameters) == type(None):
            parameters = self.parameters
        assert len(parameters) == len(self.parameters), 'Shape of parameters is incorrect ({} should be {}).'.format(len(parameters),len(self.parameters))

        time_tensor = np.tile(time,self.n_metabolites).reshape(self.n_metabolites,-1)
        tmp_n_timepoints = self.n_timepoints
        tmp_parameters = self.parameters
        self.n_timepoints=len(time)
        self.parameters = np.array(parameters)
        y = self._fun(time_tensor,self._get_tensor_parameters())
        self.n_timepoints=tmp_n_timepoints
        self.parameters = tmp_parameters
        return y
    
    def get_C_df(self,time=None,parameters=None):
        '''
        Returns pandas.DataFrame with C from the Equation M = C * V_sweat.
        self.parameters IS NOT updated.
        self.time       IS NOT updated.
        -
        Input
        time           None or numpy.ndarray of time points for which M is calculated. If None self.time is used.
        parameters     None or parameters as floats. Lists or numpy.ndarrays lead to errors down the line. If None self.parameters are used.
        -
        Output
        y              pandas.DataFrame of calculated C values and sweat volumes of shape (self.n_metabolites + 1, self.n_timepoints).
                       Sweat volumes are NOT returned!
        '''
        if type(time) == type(None):
            time = np.array(self.time)
        if type(parameters) == type(None):
            parameters = np.array(self.parameters)
        assert len(parameters) == len(self.parameters), 'Shape of parameters is incorrect ({} should be {}).'.format(len(parameters),len(self.parameters))
        assert self._has_metabolite_names, 'Model metabolite names are not set (see self.set_metabolite_names).'
        C_tensor = self.get_C_tensor(time=time,parameters=parameters)
        C_df = pd.DataFrame(index=np.arange(len(time)),dtype=float,columns=['time']+list(self.metabolite_names))
        C_df.loc[:,'time'] = time
        for i, metabolite in enumerate(self.metabolite_names):
            C_df.loc[:,metabolite] = C_tensor[i,:]
        return C_df
            
    def _optimize(self,seed):
        '''
        Internal function that optimizes the model to given measured data.
        Measured data and bounds have to be parsed with self.set_measured_data() and 
        self.set_bounds().
        -
        Input
        seed    Seed variable for np.random.seed(seed)
        -
        Output
        array   Numpy array of shape (len(self.paramters)+1) of parameters + loss of fitted model.
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
                                             absolute_sigma = True,
                                             p0      = p0,
                                             bounds  = (self.lower_bounds,self.upper_bounds),
                                             method  = 'trf',
                                             max_nfev= 1000,
                                             loss    = self._loss_function,
                                             tr_solver = 'exact',
                                            )
            loss = self.loss
            rho  = self.rho
            return rho, np.concatenate([parameters,[loss]])
        except (RuntimeError) as e:
            return np.zeros((3,len(self.measured_data)))*np.nan, np.concatenate([self.lower_bounds*np.nan,[np.inf]])
            
    
    def optimize_monte_carlo(self,n_replicates,n_cpu):
        '''
        Function that optimizes the model to given measured data.
        Measured data and bounds have to be parsed with self.set_measured_data() 
        and self.set_bounds().
        -
        Input
        n_replicates    Int. Number of monte carlo replicates to fit.
        n_cpu           Int. Number of CPUs to use for multiprocessing.
        -
        Output
        array           Numpy array of shape (len(self.paramters)+1,n_replicates) of parameters
                        + loss of fitted model times the number of replicates.

        '''
        
        assert self._has_measured_data, 'No measured data parsed.'
        assert self._has_bounds, 'No bounds parsed.'
        n_cpu = np.min([n_replicates,n_cpu])
        _input = list(range(n_replicates))
        
        # multiprocessing 
        _rho = []
        _output = []
        with Pool(processes = n_cpu) as p:
            for _ in tqdm.tqdm(p.imap_unordered(self._optimize,_input),total=n_replicates):
                _output.append(_[1])
                _rho.append(_[0])
        
        _output = np.array(_output)
        r = len(self.parameters)+1
        best_parameter = _output[np.argmin(_output[:,-1]),-r:-1]
        self.set_parameters(best_parameter)
        self._is_optimized = True
        self.loss = np.min(_output[:,-1])
        self.rho = _rho[np.argmin(_output[:,-1])]
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
        self.loss = np.sum(np.abs(rho[0]))
        self.rho = rho
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
        self.loss = np.sum(np.abs(rho[0]))
        self.rho = rho
        return rho
    
    def cauchy_loss(self,absolute_error):
        '''
        Takes array of absolute error. Cauchy loss as implemented in SciPy is calculated.
        '''
        z = absolute_error
        rho = np.empty((3,len(z)))
        rho[0] = np.log1p(z)
        t = 1 + z
        rho[1] = 1 / t
        rho[2] = -1 / t**2
        self.loss = np.sum(np.abs(rho[0]))
        self.rho = rho
        return rho

    def info(self):
        '''
        Convenience function that returns a pd.DataFrame with some overview information about self.
        -
        Output
        DataFrame    pd.DataFrame
        '''
        model_properties = {
            'n_metabolites':[self.n_metabolites],
            'n_timepoints':[self.n_timepoints],
            'pkm_fun':[self.pkm_fun_name],
            'trans_fun':[self.trans_fun_name],
            'parameters':[len(self.parameters)],
            'bounds':[self._has_bounds],
            'measured data':[self._has_measured_data],
            'metabolite names':[self._has_metabolite_names],
            'is optimized':[self._is_optimized],
            'optimization loss':[self.loss]
        }
        return pd.DataFrame(model_properties,index=['PKM Model']).transpose()

    
# MIX model
class MIX_model(PKM_model):
    def __init__(self,time,n_metabolites,pkm_fun,scale_fun,trans_fun):
        '''
        Initialization.
        -
        Input
        time             numpy.ndarray of time points of measurements.
        n_metabolites    int. number of metabolites measured.
        pkm_fun          'bateman' or 'fun_1'. Function type used for PKM. 'fun_1' is a special case of 'bateman', where ka = ke.
        scale_fun        'standard' or 'mean'. Function that scales PQN.
        trans_fun        'log10' or 'none'. Function that transforms measured data.
        -
        Output
        MIX_model class
        '''
        
        super().__init__(time,n_metabolites,pkm_fun,trans_fun)
        self.sigma = np.ones((self.n_metabolites+1)*self.n_timepoints)
        if scale_fun == 'standard':
            self.scale_fun = standard_scale
            self.scale_fun_name = scale_fun
        elif scale_fun == 'mean':
            if trans_fun == 'log10':
                self.scale_fun = log_mean_scale
                self.scale_fun_name = 'log_mean'
            elif trans_fun == 'none':
                self.scale_fun = mean_scale
                self.scale_fun_name = scale_fun
        else:
            raise ValueError('scale_fun has to be "standard" or "mean".')
        
    def set_measured_data(self,measured_data,pqn_data):
        '''
        Set measured data for model optimization (equal to M in M = C * V_sweat). It is parsed into the 
        scipy.optimize.curve_fit function and the model will be optimized to be as close as possible to it. 
        -
        Input
        measured_data    Flattened array of measured data of shape (self.n_timepoints * self.n_metabolites).
        '''
        assert len(measured_data) == self.n_metabolites*self.n_timepoints, 'Shape of measured_data is incorrect ({} should be {}).'.format(len(measured_data),self.n_metabolites*self.n_timepoints)
        assert len(pqn_data) == self.n_timepoints, 'Shape of pqn_data is incorrect ({} should be {}).'.format(len(pqn_data),self.n_timepoints)
        self.measured_data = np.concatenate([self.trans_fun(measured_data),self.scale_fun(self.trans_fun(pqn_data))])
        self._has_measured_data = True
        
    def set_sigma(self,sigma):
        '''
        Set sigma for model optimization. It is parsed into the scipy.optimize.curve_fit function.
        There the weighted error residuals are calculated according to chisq = sum((r / sigma) ** 2). I. e. sigma is the reciprocal of lambda.
        - 
        Input
        sigma    List or array of shape (self.n_timepoints * self.n_metabolites + 1).
        '''
        assert len(sigma) == self.n_timepoints*(self.n_metabolites+1), 'Shape of sigma is incorrect ({} should be {}).'.format(len(sigma),self.n_timepoints*(self.n_metabolites+1))
        self.sigma = sigma
        
    def fit(self,time,*parameters):
        '''
        Returns flattened, transformed M from the Equation M = C * V_sweat concatenated to the scaled, transformed sweat volume array.
        self.parameters IS     updated.
        self.time       IS NOT updated.
        -
        Input
        time           numpy.ndarray of time points for which M is calculated.
        *parameters    Parameters as floats. Lists or numpy.ndarrays lead to errors down the line.
        -
        Output
        y              numpy.ndarray of calculated and concatenated self.trans_fun(M) and self.scale_fun(self.trans_fun(V_sweat)) values.
        '''
        assert len(parameters) == len(self.parameters), 'Shape of parameters is incorrect ({} should be {}).'.format(len(parameters),len(self.parameters))
        
        self.parameters = np.array(parameters)
        sweat_volumes = self.get_sweat_volumes()
        sweat_volumes_tensor = np.tile(sweat_volumes,self.n_metabolites).reshape(self.n_metabolites,self.n_timepoints)
        y1 = self._fun(self._time_tensor,self._get_tensor_parameters())*sweat_volumes_tensor
        y  = np.concatenate([self.trans_fun(y1.flatten('F')),self.scale_fun(self.trans_fun(sweat_volumes))])
        return y
    
    def plot(self,time,*parameters):
        '''
        Returns flattened C from the Equation M = C * V_sweat concatenated to the sweat volume array.
        self.parameters IS NOT updated.
        self.time       IS NOT updated.
        -
        Input
        time           numpy.ndarray of time points for which M is calculated.
        *parameters    Parameters as floats. Lists or numpy.ndarrays lead to errors down the line.
        -
        Output
        y              numpy.ndarray of calculated C values.
        '''
        assert len(parameters) == len(self.parameters), 'Shape of parameters is incorrect ({} should be {}).'.format(len(parameters),len(self.parameters))
        time_tensor = np.tile(time,self.n_metabolites).reshape(self.n_metabolites,-1)
        tmp_n_timepoints = self.n_timepoints
        tmp_parameters = self.parameters
        self.n_timepoints=len(time)
        self.parameters = np.array(parameters)
        y1 = self._fun(time_tensor,self._get_tensor_parameters())
        y2 = self.get_sweat_volumes()
        y  = np.concatenate([self.trans_fun(y1.flatten('F')),self.trans_fun(y2)])
        self.n_timepoints=tmp_n_timepoints
        self.parameters = tmp_parameters
        return y
    
    def get_M_tensor(self):
        '''
        Returns unflattened, transformed M from the Equation M = C * V_sweat and the scaled, transformed sweat volume array.
        self.parameters IS NOT updated.
        self.time       IS NOT updated.
        -
        Input
        None
        -
        Output
        y              numpy.ndarray of calculated self.trans_fun(M) and self.scale_fun(self.trans_fun(V_sweat)) values.
        '''
        time = np.array(self.time)
        parameters = np.array(self.parameters)
        sweat_volumes = self.get_sweat_volumes()
        sweat_volumes_tensor = np.tile(sweat_volumes,self.n_metabolites).reshape(self.n_metabolites,self.n_timepoints)
        y1 = self._fun(self._time_tensor,self._get_tensor_parameters())*sweat_volumes_tensor
        y  = np.vstack([self.trans_fun(y1),self.scale_fun(self.trans_fun(sweat_volumes))])
        return y
    
    def get_M_df(self):
        '''
        Returns DataFrame with unflattened, transformed M from the Equation M = C * V_sweat and the scaled, transformed sweat volume array.
        self.parameters IS NOT updated.
        self.time       IS NOT updated.
        -
        Input
        None
        -
        Output
        y              pandas.DataFrame of shape (self.n_metabolites + 2, self.n_timepoints).
        '''
        assert self._has_metabolite_names, 'Model metabolite names are not set (see self.set_metabolite_names).'
        M_tensor = self.get_M_tensor()
        M_df = pd.DataFrame(index=np.arange(self.n_timepoints),dtype=float,columns=['time','V_sweat']+list(self.metabolite_names))
        M_df.loc[:,'time'] = self.time
        M_df.loc[:,'V_sweat'] = M_tensor[-1]
        for i, metabolite in enumerate(self.metabolite_names):
            M_df.loc[:,metabolite] = M_tensor[i]
        return M_df
    
    def get_C_tensor(self,time=None,parameters=None):
        '''
        Returns C from the Equation M = C * V_sweat concatenated to the sweat volume array.
        self.parameters IS NOT updated.
        self.time       IS NOT updated.
        -
        Input
        time           None or numpy.ndarray of time points for which M is calculated. If None self.time is used.
        parameters     None or parameters as floats. Lists or numpy.ndarrays lead to errors down the line. If None self.parameters are used.
        -
        Output
        y              numpy.ndarray of calculated C values and sweat volumes of shape (self.n_metabolites, self.n_timepoints).
                       Sweat volumes are NOT returned!
        '''
        if type(time) == type(None):
            time = self.time
        if type(parameters) == type(None):
            parameters = self.parameters
        assert len(parameters) == len(self.parameters), 'Shape of parameters is incorrect ({} should be {}).'.format(len(parameters),len(self.parameters))
        time_tensor = np.tile(time,self.n_metabolites).reshape(self.n_metabolites,-1)
        tmp_n_timepoints = self.n_timepoints
        tmp_parameters = self.parameters
        self.n_timepoints=len(time)
        self.parameters = np.array(parameters)
        y = self._fun(time_tensor,self._get_tensor_parameters())
        self.n_timepoints=tmp_n_timepoints
        self.parameters = tmp_parameters
        return y
    
    def max_linear_loss(self,absolute_error):
        '''
        Takes array of absolute error, calculates relative error. From the maximum of both Linear loss as implemented in SciPy is calculated.
        '''
        # Back-scale scaled PQN error
        if self.scale_fun_name == 'standard':
            # Because the error is the sum of squares, the std(V_sweat) also has to be squared, i.e. var(V_sweat).
            absolute_error[-self.n_timepoints:] = absolute_error[-self.n_timepoints:]*np.var(self.trans_fun(self.get_sweat_volumes()))

        # get true values
        y = self.plot(self.time,*self.parameters)
        relative_error = np.divide(absolute_error, y, out=absolute_error.copy(), where=y!=0)
        # maximum error
        z = np.maximum(absolute_error,relative_error)
        rho = np.zeros((3,len(z)))
        rho[0] = z
        rho[1] = np.ones(len(z))
        self.loss = np.sum(np.abs(rho[0]))
        self.rho = rho
        return rho
    
    def max_cauchy_loss(self,absolute_error):
        '''
        Takes array of absolute error, calculates relative error. From the maximum of both Cauchy loss as implemented in SciPy is calculated.
        '''
        # Back-scale scaled PQN error
        if self.scale_fun_name == 'standard':
            # Because the error is the sum of squares, the std(V_sweat) also has to be squared, i.e. var(V_sweat).
            absolute_error[-self.n_timepoints:] = absolute_error[-self.n_timepoints:]*np.var(self.trans_fun(self.get_sweat_volumes()))
            
        # get true values
        y = self.plot(self.time,*self.parameters)
        relative_error = np.divide(absolute_error, y, out=absolute_error.copy(), where=y!=0)
        # maximum error
        z = np.maximum(absolute_error,relative_error)
        rho = np.empty((3,len(z)))
        rho[0] = np.log1p(z)
        t = 1 + z
        rho[1] = 1 / t
        rho[2] = -1 / t**2
        self.loss = np.sum(np.abs(rho[0]))
        self.rho = rho
        return rho
        
    def cauchy_loss(self,absolute_error):
        '''
        Takes array of absolute error. Cauchy loss as implemented in SciPy is calculated.
        '''
        
        # Back-scale scaled PQN error
        if self.scale_fun_name == 'standard':
            # Because the error is the sum of squares, the std(V_sweat) also has to be squared, i.e. var(V_sweat).
            absolute_error[-self.n_timepoints:] = absolute_error[-self.n_timepoints:]*np.var(self.trans_fun(self.get_sweat_volumes()))
            
        z = absolute_error
        rho = np.empty((3,len(z)))
        rho[0] = np.log1p(z)
        t = 1 + z
        rho[1] = 1 / t
        rho[2] = -1 / t**2
        self.loss = np.sum(np.abs(rho[0]))
        self.rho = rho
        return rho
        
    def info(self):
        '''
        Convenience function that returns a pd.DataFrame with some overview information about self.
        -
        Output
        DataFrame    pd.DataFrame
        '''
        model_properties = {
            'n_metabolites':[self.n_metabolites],
            'n_timepoints':[self.n_timepoints],
            'pkm_fun':[self.pkm_fun_name],
            'trans_fun':[self.trans_fun_name],
            'scale_fun':[self.scale_fun_name],
            'parameters':[len(self.parameters)],
            'bounds':[self._has_bounds],
            'measured data':[self._has_measured_data],
            'metabolite names':[self._has_metabolite_names],
            'is optimized':[self._is_optimized],
            'optimization loss':[self.loss]
        }
        return pd.DataFrame(model_properties,index=['MIX Model']).transpose()
