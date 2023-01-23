import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
from scipy.integrate import trapz
from scipy import stats
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["font.sans-serif"] = "Arial"
import mpl_toolkits
matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["font.sans-serif"] = "Arial"
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import pwlf

#os.chdir("C:\\Users\\atstr\\Box\\Lab\\Murugan\Kernel Code") #make cwd the kernel code, if on desktop
os.chdir("/Users/adamstrupp/Library/CloudStorage/Box-Box/Lab/Murugan/Kernel_Code") #if on laptop
 
class ActiveFlowNetworks:

    def __init__(self, dt = 1e-3, tau = 1):

        # whenever you create an instance of a python class (that's what
        # Sim = ActiveFlowNetworks() does), the __init__ function automatically
        # gets run. For us, this sets up variables which we will want later on.
        self.tau =tau
        self.dt = dt
 # original mesh size = 1/dt             mesh size = length of array 
        self.kernel_mesh_size = int(1/self.dt) # tau second worth of mesh, currently 5000 = 1/0.0002 #mesh size increases as foctor of tau now
        self.kernel = self._initialize_ecoli_kernel() #defaults to ecoli
        self.task_id=None
        self.max_task_id=None
    
    def kernel_integrator(self): #returns integral and integral of absolute value
        sum = 0
        absum = 0
        for i in range(len(self.kernel)):
            absum += np.abs(self.kernel[i])
            sum += self.kernel[i]
        integral = sum * self.dt
        abintegral = absum * self.dt
        return integral, abintegral 


    def _initialize_ecoli_kernel(self):

        #this function returns a kernel which approximates the
        #ecoli chemotactic response kernel.
        #see Celani and Vergassola 2010 for a description of the various constants.

        #you can play around with beta, D, and the scaling of t
        #and see what the various shapes look like.
        B = 20

        n = (B**3) #normalize to account for tau scaling and fiorst moment
    

        #normalization_factor = 140 #calculated to make integral of absolute value = 2

        #beta = 1
        #D = 1/6

        #tau = 1/3/D
        #lam = 4*(1+3*D*tau)/3/tau

        #instead of including tau here and 1/tau in the kernel function I omitted both
        t = np.arange(self.kernel_mesh_size)/self.kernel_mesh_size  #the ten makes it so the kernel equation has large enough range to show proper shape, 
        #but does not change the actual "time" which is instead measured by number of timesteps
        # also same as multiplying by 10dt(it doesnt matter how big the numbers are just how long the list is.list [10/meshsize), (20/meshsize), ... 10] has meshsize number of elements and goes 1 to 10

        #kernel = normalization_factor*beta*lam*np.exp(-lam*t)*(lam*t-(lam*t)**2/2)
        
        kernel = n*np.exp(-B*t)*(t-(B/2)*(t**2)) #no tau means its just sampled denser #height adjusted for tau

        return kernel


    def _initialize_uneven_ecoli_kernel(self,I):

        #this function returns a kernel which approximates the
        #ecoli chemotactic response kernel.
        #see Celani and Vergassola 2010 for a description of the various constants.

        #you can play around with beta, D, and the scaling of t
        #and see what the various shapes look like.
        B = 20

        A = 3*I*(B**2) + B**3 #normalize to account for tau scaling and fiorst moment
        C = (B**2+2*B*I)/(6*I+2*B)
    

        #normalization_factor = 140 #calculated to make integral of absolute value = 2

        #beta = 1
        #D = 1/6

        #tau = 1/3/D
        #lam = 4*(1+3*D*tau)/3/tau

        #instead of including tau here and 1/tau in the kernel function I omitted both
        t = np.arange(self.kernel_mesh_size)/self.kernel_mesh_size  #the ten makes it so the kernel equation has large enough range to show proper shape, 
        #but does not change the actual "time" which is instead measured by number of timesteps
        # also same as multiplying by 10dt(it doesnt matter how big the numbers are just how long the list is.list [10/meshsize), (20/meshsize), ... 10] has meshsize number of elements and goes 1 to 10

        #kernel = normalization_factor*beta*lam*np.exp(-lam*t)*(lam*t-(lam*t)**2/2)
        
        kernel = A*np.exp(-B*t)*(t-(C)*(t**2)) #no tau means its just sampled denser #height adjusted for tau

        return kernel

    def _initialize_step_kernel(self):

        #this function initializes a step function kernel (+const to -const)
        #you can play around with the normalization const
        #note that the second line assumes an even number of mesh points
        scaling_factor = 4/self.tau #calculated so convolution returns actual slope. note abs_AUC now = 4
        kernel = np.ones(self.kernel_mesh_size)
        kernel[len(kernel)//2:] = -1 #flips sign of second half of list
        kernel *= scaling_factor  # sets the elemetns to 4

        return kernel

    def _initialize_sin_kernel(self):

        #this function initializes a sine kernel
        #note that it is one element longer than the other kernels
        scaling_factor = 2* np.pi /(self.tau**2) #did integration to normalize to pi/(2* deltat^2)
        kernel = scaling_factor*np.sin(np.arange(self.kernel_mesh_size+1)/self.kernel_mesh_size*2*np.pi) #max element is 2pi so that sin does a full cycle. it is one longer because arrange isnt inclusive and he wanted the last to actually be 2pi

        return kernel

    def construct_external_signal(self,free_value, clamp_value, fast_time, slow_time): #10>slow time=total length> fast time
        #fast and slow time need to be multiples of dt
        #make x list with the same mesh size per second as the kernels but lasting more seconds = slow_time
        tlist = np.arange(slow_time/(self.dt))*self.dt #still goes to slow time #tau shorter list but goes to same value#now in terms of dt to ingnore tau #time step dt over slow_time seconds, max value slow_time-dt
        #has more timesteps than kernels so that kernels can pan over different sections (kernel =1s here)

        # Make skeleton x and y lists to be interpolated
        x = [0,fast_time, slow_time] #could make robust to time values not multiple of dt by rounding to nearest dt
        y = [free_value, clamp_value, free_value]
        interpolated_sawtooth = interp1d(x,y)
        sawtooth = interpolated_sawtooth(tlist) #discrete sawtooth that goes 0 to slow_time

        #plot it

        return sawtooth

    def compute_period_weight_update(self, free_value, clamp_value, fast_time, slow_time, nonlinearity_threshold, nonlinearity_style = "linear_threshold", downswing ='yes'):


        #to proceed with this calculation, you will need to write a function
        #which will construct the external signal for you. This will be an
        #array of values that quickly goes from the free_value to the clamp_value
        #and then slowly back to the free_value.
        #you might also want to pass the fast and slow rates of the sawtooth here too.
        #make sure that the timestep between successive points in your external signal
        #matches the timestep dt of the kernel grid!
        #you might find the scipy interp1d function useful here, but I'm not sure.
        if downswing == 'yes':
            external_signal = self.construct_external_signal(free_value, clamp_value, fast_time, slow_time)
        
        if downswing == 'no':
            external_signal = self.construct_external_signal(free_value, clamp_value, fast_time, slow_time = fast_time + self.dt)[:-1]

        #this line of code performs the convolution of the kernel and
        #the external signal. it will probably be useful for you to look
        #at the documentation of this function, which is in the scipy library.
        if downswing == 'yes':
            dsdt = convolve1d(external_signal,self.kernel,mode='wrap')*self.dt #larger distance between points for larger tau
        #since each element of the convolution is a dot product of mesh_size elements, I should divide by mesh size (multiply by dt) to take average value
        # has same len as external_signal with all kernels (even sin)
        #print("length of dsdt =", len(dsdt))
        #print("length of sawtooth =", len(external_signal))
        if downswing == 'no':
            dsdt = convolve1d(external_signal,self.kernel,mode='mirror')*self.dt
       
       #the kernels run right to left but convolving reverses this order
        # wrap actually applies to the inout which is good. inout should wrap

       #what I want is a list of values the same length as the sawtooth that represents the derivative at each point
       #the kernel convolved around a section of it centered at each place
       
       # can use constant mode to fill zero outside kernel range
       
        # may not want wrap mode as we want kernel to not repeat (measuring multiple points at once, but istead sall off (not step function but square function))
        #you can use the following lines of code which were commented out
        #if you want to test what happens with the code if you just take the
        #fast part of the externel signal, just as a check. you will have to
        #set the hack_cutoff variable for this.
        #hack_cutoff = 
        #dk = trapz(dkdt[0:hack_cutoff],self.eval_grid[0:hack_cutoff],axis=0)

        #this is the line of code that currently implements the nonlinearity.
        #try replacing it with other things! if the computation becomes unwieldy,
        #you can always separate it out into a different function.
        g_dsdt = self.apply_nonlinearity(dsdt, nonlinearity_threshold, nonlinearity_style)
        
        #this code computes the integral of the (non-linearly transformed) derivative.
        #you might consider playing around with the normalization of this integral
        #for instance, dividing by the AUC of the absolute value of the kernel.
        #assumes constant mesh for integral of size dt.
        ds = trapz(g_dsdt,dx=self.dt) #longer steps

        return ds

    def apply_nonlinearity(self, dsdt, threshold = 1, style="linear_threshold"):
        self.rate_cutoff = threshold
        if style == "linear_threshold":
            dsdt[np.abs(dsdt) < self.rate_cutoff] = 0 #discontinuous linear kernel (closest to eq prop but requires threshold to bisec rates)
        if style == "cubic":
            for i in range(len(dsdt)):
                dsdt[i] = ((dsdt[i]**3))/self.rate_cutoff #scaled down cubic kernel

        
        return dsdt
    def integral_scatterplot(self, free_value, height_range = 50, nonlinearity_threshold = 2, fast_time = 5, slow_time = 15): #pass free value, iterate over clamped values
        I_list = []
        for i in range(height_range):
            I_list.append(self.compute_period_weight_update(free_value = free_value, clamp_value = (free_value + i), fast_time = fast_time, slow_time = slow_time, nonlinearity_threshold = nonlinearity_threshold, nonlinearity_style = "linear_threshold"))
        return I_list
    #plots integral vs clamped-free

    def r_value (self, x,y):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        return r_value

    def r_vs_g(self, fast_time, slow_time, free_value): #plots r of (I vs 5-4) vs g threshold
        g_list = []
        r_list = []
        for i in range(500): #try many values of 3, 
            g_list.append(i/100) #record g
            I_list = self.integral_scatterplot(free_value = free_value, height_range = 20, nonlinearity_threshold = i/100) #get intragrals as function of 5
            r_list.append(self.r_value(list(range(20)), I_list)) #get r of I vs 5-4 graph
        fig, ax = plt.subplots()
        ax.set(xlabel = "Nonlinearity Threshold", ylabel = "r-value", title = ("For fast_time", fast_time , "slow time", slow_time, "free value", free_value))
        ax.plot(g_list,r_list)
        plt.show()
        return 
        
    def variation_of_parameters(self, fv): # does not vary free_value, kernel type, g type
        fast_times = []
        slow_times = []
        thresholds = []
        r_values = []
        for fast_time in range (20): ####### 20
            print("starting iteration ", fast_time +1 , "of 20")
            for extra_time in range (50): ##50
                for th in range (50): #####50
                    slow_time = (fast_time + extra_time)/4
                    threshold = th/5
                    I_list = self.integral_scatterplot(free_value = fv, nonlinearity_threshold = threshold, fast_time = fast_time/4, slow_time = slow_time)
                    x = list(range(len(I_list)))
                    r = self.r_value(x,I_list)
                    fast_times.append(fast_time/2)
                    slow_times.append(slow_time)
                    thresholds.append(threshold)
                    r_values.append(r)
        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")
        # Creating plot
        #cmap = matplotlib.colors.ListedColormap(sbn.color_palette("husl", 256).as_hex())

        ax.scatter(fast_times, slow_times, thresholds, s=40, c=r_values, marker='.', cmap='seismic', alpha=.2)

        ax.set(xlabel = "fast time", ylabel = "slow time", zlabel = "g threshold", title = ("For free value" + str(fv)))

        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
        cmap = matplotlib.colors.ListedColormap('seismic')
    
        #fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
        # fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal', label='Some Units')
        plt.show()

    def change_mesh_size(self, factor, kernel= 'ecoli'):
        
       ###
        if kernel == "sin":
            self.kernel = self._initialize_sin_kernel()
        if kernel == "step":
            self.kernel = self._initialize_step_kernel()
        if kernel == "ecoli":
            self.kernel = self._initialize_ecoli_kernel()

    def vary_fv_and_mesh(self, height_range = 50, nonlinearity_threshold = 0.75, fast_time = 5, slow_time = 20, kernel= 'ecoli'):
        label_list = ["sb", "vr", "1g"]
        mesh_list = [0.01,0.5,20]
        for fv in (0,10,250):
            free_value = fv
            fig, ax = plt.subplots()
            ax.set(xlabel = "Sawtooth Height", ylabel = "Convolution Integral", title = ("free value", free_value)  )
            for i in (0,1,2):
                self.change_mesh_size(mesh_list[i], kernel)
                I_list = self.integral_scatterplot(free_value = free_value, height_range = height_range, nonlinearity_threshold = nonlinearity_threshold, fast_time = fast_time, slow_time = slow_time)
                ax.plot(list(range(height_range)), I_list, label_list[i], label=("mesh scaling ", mesh_list[i]))
        
            plt.legend(loc="upper left")
            savelocation =("/Users/adamstrupp/Library/CloudStorage/Box-Box/Lab/Murugan/Kernel Code/mesh and_free_value graphs" )
            plt.savefig("test", format = "png")
            plt.show()

    def vary_just_mesh(self, nonlinearity_threshold = 1, fast_time = 5, slow_time = 20, kernel= 'ecoli'):
        free_value = 0
        fig, ax = plt.subplots()
        mesh_list = []
        I_list = []
        ax.set(xlabel = "Mesh Size", ylabel = "Convolution Integral", title = 'Effect of Mesh Size on integral of sawtooth with height 10' )
        for i in range(20):
            self.change_mesh_size((i+1)/2, kernel)
            mesh_list.append(self.kernel_mesh_size)
            I = self.compute_period_weight_update(free_value, 10, fast_time, slow_time, nonlinearity_threshold, nonlinearity_style = "linear_threshold")
            I_list.append(I)

        ax.plot(mesh_list, I_list)
        
        plt.legend(loc="upper left")
        #savelocation =("/Users/adamstrupp/Library/CloudStorage/Box-Box/Lab/Murugan/Kernel Code/mesh and_free_value graphs" )
        #plt.savefig("test", format = "png")
        plt.show()

    def tau_vs_error(self): #try with other kernels
        taulist = []
        error_list = []
        for i in range (1,50):
            self.tau = (i)
            self._initialize_ecoli_kernel() #needs to shrink for tau
            I = self.compute_period_weight_update(0,10,3,13,2)
            A = 10
            error = A-I
            taulist.append(self.tau)
            error_list.append(error)

        plt.plot(taulist, error_list)
        plt.xlabel('Tau')
        plt.ylabel('Error')
        plt.title("Tau Vs Integration Error w/ ecoli kernel")
        plt.show()
    
    def times_from_rates(self, fast_rate, slow_rate, free_value, clamped_value):

        fast_time = (clamped_value - free_value)/fast_rate
        slow_time = fast_time + ((clamped_value - free_value)/slow_rate)

        return fast_time, slow_time 

    def ratio_errormap(self, free_value, clamp_value):
        heatlist = [] #difference between integral value and actual free-clamped
        x_list = [] #ratios of fast rate to g_thresh
        y_list = [] #ratios of fast rate to slow rate
        fast_rate = 5
        for g in range (1,100):
            g_thresh = g/10
            for i in range (1,200):
                slow_rate = i/(200/fast_rate)
                fast_time, slow_time = self.times_from_rates(fast_rate, slow_rate, free_value, clamp_value)
                integral = self.compute_period_weight_update(free_value, clamp_value, fast_time, slow_time, g_thresh, nonlinearity_style = "linear_threshold")
                actual_difference = clamp_value - free_value
                error = integral - actual_difference
                y_list.append(slow_rate/fast_rate)
                x_list.append(g_thresh/fast_rate)
                heatlist.append(abs(error))
        return x_list, y_list, heatlist
    
    def plot_errormap(self, x_list, y_list, heatlist, free_value = 0, clamp_value = 10, fast_rate = 5):
        cm = plt.cm.get_cmap('RdYlBu')
        plt.scatter(x_list, y_list, c= heatlist, s = 4, vmin = 0, vmax = max(heatlist), cmap = cm)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Absolute Value of Error', rotation =270)
        plt.xlabel("nonlinearity threshold/fast rate")
        plt.ylabel("slow rate/fast rate")
        plt.title("Kernal Error With Varying Parameters " + "FV: " + str(free_value) + " CV: " + str(clamp_value) + " FR: " + str(fast_rate))

        #plt.imshow(z)
        
        #plt.set(xlabel = "fast time / g_threshold", ylabel = "fast_time / slow time", title = ("Performance Ratios For FV" + free_value + "CV" + clamp_value))

    
        plt.show()

    def f3_p1(self): #I should probably save this data somewhere
        self.kernel = self.initialize_uneven_step(2,0.004)
        t1 = 4
        t2 = 40 #slow-fast
        theta = 3
        ds_list = []
        A_list = []
        fv_list = []
        for i in range(0,400,5):
            if i % 100 == 0 :
                print("iteration" + str(i/10))
            for j in range(0,200,50):
                fv=np.random.randint(-350,350)
                A_list.append(i+1)
                fv_list.append(fv)
                clamp_value = i+1 + fv
                ds = self.compute_period_weight_update(fv, clamp_value, fast_time = t1, slow_time = t1+t2, nonlinearity_threshold = theta, nonlinearity_style = "linear_threshold")
                ds_list.append(ds)
        print(A_list)
        cm = plt.cm.get_cmap('winter')
        #make the colored graphs
        normalize = matplotlib.colors.Normalize(vmin = 0, vmax = max(fv_list))
       
        plt.scatter(A_list, ds_list, s=15, c = fv_list, norm = normalize, cmap = cm)
        plt.title('Synapse Response to Varying Signal',fontsize =20)
        plt.xlabel('Signal Amplitide', fontsize =15)
        plt.ylabel('Synapse Output', fontsize =15)
        #plt.xticks(range(0,151,50)) #make big ticks
        #plt.yticks(range(0,101,50))
        plt.tick_params(axis='both', which='major', labelsize=10, width=4, length=10)
        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=normalize, cmap=cm))
        cbar.set_label('Free Value', rotation=270, fontsize  = 15)
        plt.xlim(0,250)
        plt.ylim(-10,150)
        plt.show()
        # with open("f3p1A_list.txt", 'w') as f:
        #     for a in A_list:
        #         f.write(str(a) + '\n')
        # with open("f3p1ds_list.txt", 'w') as f:
        #     for ds in ds_list:
        #         f.write(str(ds) + '\n')
        return A_list, ds_list

    def speed_f3_p1(self): #I should probably save this data somewhere
            self.kernel = self.initialize_uneven_step(2,0.03)
            theta = 25
            fv = 300
            for time in [1,2,3,4,5]:
                t1 = time
                t2 = 4*t1
                ds_list = []
                A_list = []
                for i in range(-340,340,20):
                    A_list.append(i+1)
                    clamp_value = i+1 + fv
                    ds = self.compute_period_weight_update(fv, clamp_value, fast_time = t1, slow_time = t1+t2, nonlinearity_threshold = 50*t1, nonlinearity_style = "linear_threshold")
                    ds_list.append(ds)
                
        
                plt.plot(A_list, ds_list, label = t1)
            plt.title('Synapse Response to Noisy Training',fontsize =20)
            plt.xlabel('Signal Amplitide', fontsize =15)
            plt.ylabel('Synapse Output', fontsize =15)
            #plt.xticks(range(0,151,50)) #make big ticks
            #plt.yticks(range(0,101,50))
            plt.tick_params(axis='both', which='major', labelsize=10, width=4, length=10)

            plt.legend(title = 'Time', loc = 'upper left')
            plt.show()

    def line_f3_p1(self): #I should probably save this data somewhere
            self.kernel = self.initialize_uneven_step(2,0.03)
            t1 = 1
            t2 = 4 #slow-fast
            theta = 25
            ds_list = []
            A_list = []
            for j in [-400,0,400]:
                ds_list = []
                A_list = []
                fv=j
                for i in range(-340,340,20):
                    A_list.append(i+1)
                    clamp_value = i+1 + fv
                    ds = self.compute_period_weight_update(fv, clamp_value, fast_time = t1, slow_time = t1+t2, nonlinearity_threshold = theta, nonlinearity_style = "linear_threshold")
                    ds_list.append(ds)
                
        
                plt.plot(A_list, ds_list, label = fv)
            plt.title('Synapse Response to Noisy Training',fontsize =20)
            plt.xlabel('Signal Amplitide', fontsize =15)
            plt.ylabel('Synapse Output', fontsize =15)
            #plt.xticks(range(0,151,50)) #make big ticks
            #plt.yticks(range(0,101,50))
            plt.tick_params(axis='both', which='major', labelsize=10, width=4, length=10)

            plt.legend(title = 'Free Value', loc = 'upper left')
            plt.show()
    
    def fixav_line_f3_p1(self): #I should probably save this data somewhere
            self.kernel = self.initialize_uneven_step(4,0.03)
            t1 = 10
            t2 = 100 #slow-fast
            theta = 5
            ds_list = []
            A_list = []
            for j in [-20,0,20]:
                ds_list = []
                A_list = []
                average=j
                for i in range(-258,258,10):
                    A_list.append(i+1/10)
                    print(i)
                    deviation = i/2
                    #clamp_value = i+1 + fv
                    ds = self.compute_period_weight_update(average - deviation, average + deviation, fast_time = t1, slow_time = t1+t2, nonlinearity_threshold = theta, nonlinearity_style = "linear_threshold")
                    ds_list.append(ds/i)
                
        
                plt.plot(A_list, ds_list, label = average)
            plt.title('Synapse Response to Noisy Training',fontsize =20)
            plt.xlabel('Signal Amplitide/t1*theta', fontsize =15)
            plt.ylabel('Synapse Output/Input', fontsize =15)
            #plt.xticks(range(0,151,50)) #make big ticks
            #plt.yticks(range(0,101,50))
            plt.tick_params(axis='both', which='major', labelsize=10, width=4, length=10)

            plt.legend(title = 'Average Value', loc = 'lower left')
            plt.savefig('b.pdf', format = 'pdf', bbox_inches="tight")

            plt.show()
        

    def timef3_p1(self): #I should probably save this data somewhere
        self.kernel = self.initialize_uneven_step(2,0.004)
        #t1 = 4
        t2 = 20 #slow-fast
        theta = 3
        ds_list = []
        A_list = []
        tf_list = []
        fv =500
        for i in range(0,200,5):
            if i % 100 == 0 :
                print("iteration" + str(i/10))
            for j in range(6):
                #tf = j/2 + 1
                tf = np.random.randint(1,10)
                A_list.append(i+1)
                tf_list.append(tf)
                clamp_value = fv+i+1
                ds = self.compute_period_weight_update(fv, clamp_value, fast_time = tf, slow_time = 30, nonlinearity_threshold = 5, nonlinearity_style = "linear_threshold", downswing = 'yes')
                ds_list.append(ds)
        cm = plt.cm.get_cmap('winter')
        #make the colored graphs
        normalize = matplotlib.colors.Normalize(vmin = 0, vmax = max(tf_list))
       
        plt.scatter(A_list, ds_list, s=15, c = tf_list, norm = normalize, cmap = cm)
        plt.title(f'Synapse Response to Noisy Training',fontsize =20)
        plt.xlabel('Signal Amplitide', fontsize =15)
        plt.ylabel('Synapse Output', fontsize =15)
        #plt.xticks(range(0,151,50)) #make big ticks
        #plt.yticks(range(0,101,50))
        plt.tick_params(axis='both', which='major', labelsize=10, width=4, length=10)
        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=normalize, cmap=cm))
        cbar.set_label('Fast Time', rotation=270, fontsize  = 15)
        #plt.xlim(0,250)
        #plt.ylim(-10,150)
        plt.show()
        # with open("f3p1A_list.txt", 'w') as f:
        #     for a in A_list:
        #         f.write(str(a) + '\n')
        # with open("f3p1ds_list.txt", 'w') as f:
        #     for ds in ds_list:
        #         f.write(str(ds) + '\n')
        return A_list, ds_list
    
    def minimal_f3_p1(self, t1, t2): #I should probably save this data somewhere
        theta = 3
        free_value = 0
        ds_list = []
        A_list = []
        for i in range(80): #should go 0 to theta times max t2-t1 range
            A_list.append(i+1) #xstep is number here
            clamp_value = i+1
            ds = self.compute_period_weight_update( free_value, clamp_value, fast_time = t1, slow_time = t1+t2, nonlinearity_threshold = theta, nonlinearity_style = "linear_threshold")
            ds_list.append(ds)

        
        return A_list, ds_list

    def simplelines_f3_p1(self,filename): #I should probably save this data somewhere
            self.kernel = self.initialize_uneven_step(4,0.05)
            t1 = 2
            t2 = 10 #slow-fast
            theta = 10
            ds_list = []
            A_list = []
            cm = plt.cm.get_cmap('coolwarm')
            for j in [-100,-50,0,50,100]:
                ds_list = []
                A_list = []
                average=j
                for i in range(-150,150,10):
                    A_list.append(i+1/10)
                    deviation = i/2
                    #clamp_value = i+1 + fv
                    ds = self.compute_period_weight_update(average - deviation, average + deviation, fast_time = t1, slow_time = t1+t2, nonlinearity_threshold = theta, nonlinearity_style = "linear_threshold")
                    ds_list.append(ds)
                color=cm((j+100)/200)
                plt.plot(A_list, ds_list,c=color,label = average)
            plt.title('Synapse Response to Noisy Training',fontsize =20)
            plt.xlabel('Signal Amplitide', fontsize =15)
            plt.ylabel('Synapse Output', fontsize =15)
            #plt.xticks(range(0,151,50)) #make big ticks
            #plt.yticks(range(0,101,50))
            plt.tick_params(axis='both', which='major', labelsize=10, width=2, length=5)

            plt.legend(title = 'Average Value', loc = 'upper left')
            plt.savefig(f'{filename}.pdf', format = 'pdf', bbox_inches="tight")
            plt.show()
            return ds_list
        
    def Amax_f3_p1(self,filename,Am=150): #per Martin convo 11/22/22
            self.kernel = self.initialize_uneven_step(4,0) # I = 0 for now
            t1 = 1
            t2 = 9 #slow-fast
            ds_list = []
            A_list = []
            # datachunk = np.zeros(()) can save data here
            cm = plt.cm.get_cmap('coolwarm')
            for j in [0]: # see if j matters
                for Amax in [Am]:
                    theta = Amax/t2
                    ds_list = []
                    A_list = []
                    average=j
                    n=1
                    for i in range(-2*Amax,2*Amax):
                        A_list.append(i/(Amax))
                        deviation = i/(2)
                        #clamp_value = i+1 + fv
                        ds = self.compute_period_weight_update(average - deviation, average + deviation, fast_time = t1, slow_time = t1+t2, nonlinearity_threshold = theta, nonlinearity_style = "linear_threshold")
                        ds_list.append(ds/Amax)
                    color=cm((j+100)/200)
                    #plt.plot(A_list, ds_list,c=color,label = Amax)
            plt.title('updating theta with fixed Amax',fontsize =20)
            plt.xlabel('A/Amax', fontsize =15)
            plt.ylabel('Synapse Output/Amax', fontsize =15)
            #plt.xticks(range(0,151,50)) #make big ticks
            #plt.yticks(range(0,101,50))
            plt.tick_params(axis='both', which='major', labelsize=10, width=2, length=5)

            plt.legend(title = 'Average Value', loc = 'upper left')
            plt.savefig(f'{filename}.pdf', format = 'pdf', bbox_inches="tight")
            plt.show()
            Aminplus, Aminminus,breaks,slopes,x_hat,y_hat = self.piecewise_Amax(A_list,ds_list)
            plt.plot(x_hat,y_hat)
            return Aminplus, Aminminus

    def small_Amax_f3_p1(self,Am,plot='no',n=1,t1=1,t2=9,average=0): #redo with no frills for larger comp
        ds_list = []
        A_list = []
        theta = Am/t2 #pick a theta that will stop range right at Am
        n=n #try changing this #was 2 now 4, now 1
        for i in range(-6*Am,6*Am,n): #then dividing i by 4 to sample densely
            A_list.append(i/(8*(Am))) #operating with i/4 as the unit
            deviation = i/(8*(2))
            ds = self.compute_period_weight_update(average - deviation, average + deviation, fast_time = t1, slow_time = t1+t2, nonlinearity_threshold = theta, nonlinearity_style = "linear_threshold")
            ds_list.append(ds/Am)
        intercept = self.piecewise_failure_stats(A_list,ds_list)[1]
        if plot == 'yes':   #finish this 
            plt.plot(A_list,ds_list,label = Am)
            plt.show()
        # Aminplus, Aminminus = self.piecewise_Amax(A_list,ds_list)[0:2] #get ist and 2nd return values, using the linear fit
        Aminplus, Aminminus = self.first_nonzero_value(A_list,ds_list,int(np.floor((len(A_list)/2))))[0:2] #zeroindex right in the middle
        return Aminplus, intercept

    def Amax_f3_p2_cube(self):
        self.kernel = self.initialize_uneven_step(4,0)
        Adatacube = np.zeros((300,6))
        for counter in range(300):
            t1 = np.random.randint(1,10)
            t2 = np.random.randint(1,20)
            Amin = self.small_Amax_f3_p1(100,plot='no',n=1,t1=t1,t2=t2)
            Arange = (1-Amin)  
            Adatacube[counter] = [t1,t2,t1+t2,t1/t2,Amin,Arange]
        np.save('Adatacube2.npy',Adatacube)

    def expanded_Amax_f3_p2_cube(self):   #12/11/22 running to sample denser the amax_f3_p2 plot
        # self.kernel = self.initialize_uneven_step(4,0) previously ran for step kernel 


        self.kernel = self._initialize_ecoli_kernel()
        Adatacube = np.zeros((1000,3))
        for counter in range(1000):
            if counter%10==0:
                print(f'iteration_{counter}')
            t1 = (np.random.randint(1,25))/5
            t2 = (np.random.randint(1,250))/5
            Amin_ratio = self.small_Amax_f3_p1(100,plot='no',n=1,t1=t1,t2=t2) #this returns Amin/Amax for given amax
            Adatacube[counter] = [t1,t2,Amin_ratio]
        np.save('expanded_ecoli_Adatacube12_13.npy',Adatacube)

    def midway_expanded_Amax_f3_p2_cube(self,I,av):   #task_id should go 0 to 49
        total = 20 #number of points to be generated per zero indexed script to get 1000 total
        self.kernel = self._initialize_uneven_ecoli_kernel(I)
        Adatacube = np.zeros((total,4)) #
        for counter in range(total): #
            t1 = (np.random.randint(1,25))/5
            t2 = (np.random.randint(1,250))/5
            Amin_ratio, intercept = self.small_Amax_f3_p1(100,plot='no',n=1,t1=t1,t2=t2,average=av) #this returns Amin/Amax for given amax
            Adatacube[counter] = [t1,t2,Amin_ratio,intercept]
        np.save(f'midway_expanded_ecoli_cube_12_24_{self.task_id}.npy',Adatacube)

    def Amax_f3_p2_plot_0AUC(self,filename = 'ecoli_thresholded'): #used for Adatacube2 originally
        Acube = np.load('final_ecoli_0AUC_cube2.npy') #cube is [t1,t2,t1+t2,t1/t2,amin_rat,range=1-amin_rat] #was adatacube 2, now expanded cube is t1,t2,aminrat by 1000
        cm = plt.cm.get_cmap('winter')
        Acube[:,2] = [np.log(1/x) for x in Acube[:,2]] #now its amax/amin
        Acube[:,0] = [x*20 for x in Acube[:,0]] #multiple t1 by 20 to be in units of kernel decay timescale
        Acube[:,1] = [x*20 for x in Acube[:,1]] #same with t2

        indices = [i for i, x in enumerate(Acube[:,0]) if (x > 20)] #remove large t1
        Acube[indices,:] = None

        #taking just the low t1 values for diagnostic
        # for i in range(1000):
        #      if Acube[i,0] > 1:
        #         Acube[i,:] = 0        #plt.plot(Acube[:,0])kk

        #make the colored graphs     , 
        normalize = matplotlib.colors.Normalize(vmin = min(Acube[:,0]), vmax = max(Acube[:,0])) # 0 is t1, color by t1
        plt.scatter([np.log(Acube[x,1]/Acube[x,0]) for x in range(1000)],Acube[:,2], s=30, c = Acube[:,0], norm = normalize, cmap = cm)
        plt.xlabel('log(t2/t1)') #the kernel timescale is 1/20
        plt.ylabel('log(Amax/Amin)')
       # plt.plot([*range(15,100)],[np.log((x-4)/4) for x in range(15,1000)], label = 't2/min(t1)') # for x axis t1+t2
        # plt.scatter([Acube[x,1]/Acube[x,0] for x in range(10,1000)],[np.log(Acube[x,1]/4) for x in range(10,1000)], label = 't2/min(t1)') #for x axis t2/t1
        plt.plot(Acube[:,0])
        plt.legend()
        plt.title('AUC = 0, average val = 0')
        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=normalize, cmap=cm))
        cbar.set_label('t1/t_k', rotation=0, fontsize  = 15)
        # for i in [0.2,0.3,0.6,1]:
        #     plt.plot([*range(50)],[np.log((x-(i))/(i)) for x in range(50)], label = f'log(x-{i})/{i}')
        #     plt.legend() #this is the analogous plot for t2/t1 = amax/amin best case scenario for lowest t1=1, t2=t2+t1-1 #
        #for expanded, lowest t1 = 0.2, t2 = t2+t1-0.2
        plt.savefig(f'{filename}.pdf', format = 'pdf', bbox_inches="tight")

    def Amax_f3_p2_plot_int(self,filename = 'test'): #intercept is cube[:,3]
        Acube = np.load('12_24_intercept_cube_2.npy') #cube is [t1,t2,t1+t2,t1/t2,amin_rat,range=1-amin_rat] #was adatacube 2, now expanded cube is t1,t2,aminrat by 1000
        cm = plt.cm.get_cmap('winter')
        Acube[:,2] = [np.log(1/x) for x in Acube[:,2]] #now its amax/amin
        Acube[:,0] = [x*20 for x in Acube[:,0]] #multiple t1 by 20 to be in units of kernel decay timescale
        Acube[:,1] = [x*20 for x in Acube[:,1]] #same with t2

        indices = [i for i, x in enumerate(Acube[:,3]) if (x == 12 or x < 0 )] #remove errors
        Acube[indices,:] = None

        Acube[:,2] = Acube[:,1] + Acube[:,0] #replace range with total time

        # normalize = matplotlib.colors.Normalize(vmin = min(Acube[:,0]), vmax = max(Acube[:,0])) # 0 is t1, color by t1
        normalize = matplotlib.colors.Normalize(vmin = min(Acube[:,2]), vmax = max(Acube[:,2])) # color by total time

        # plt.scatter([(Acube[x,1] + Acube[x,0]) for x in range(len(Acube[:,2]))],Acube[:,3], s=30, c = Acube[:,0], norm = normalize, cmap = cm) #plotting by total t
        plt.scatter([(Acube[x,0]) for x in range(len(Acube[:,2]))],Acube[:,3], s=30, c = Acube[:,2], norm = normalize, cmap = cm) #plotting by t1

        plt.xlabel('(t1)/t_k') #the kernel timescale is 1/20
        plt.ylabel('y_int/Amax')
    
        # plt.legend()
        plt.title('AUC = 0.05, average val = 25')
        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=normalize, cmap=cm))
        cbar.set_label('t1+t2/t_k', rotation=270, fontsize  = 15)
        plt.savefig(f'{filename}.pdf', format = 'pdf', bbox_inches="tight")
    
    def Amax_f3_p2_plot_int_v2(self,filename = '1_9_t1_plot'): #running now with x axis #intercept is cube[:,3]
        Acube = np.load('12_24_intercept_cube_2.npy') #cube is [t1,t2,t1+t2,t1/t2,amin_rat,range=1-amin_rat] #was adatacube 2, now expanded cube is t1,t2,aminrat by 1000
        cm = plt.cm.get_cmap('winter')
        Acube[:,2] = [np.log(1/x) for x in Acube[:,2]] #now its amax/amin
        Acube[:,0] = [x*20 for x in Acube[:,0]] #multiple t1 by 20 to be in units of kernel decay timescale
        Acube[:,1] = [x*20 for x in Acube[:,1]] #same with t2

        indices = [i for i, x in enumerate(Acube[:,3]) if (x == 12 or x < 0 )] #remove errors
        Acube[indices,:] = None

        Acube[:,2] = Acube[:,1] + Acube[:,0] #replace range with total time

        # normalize = matplotlib.colors.Normalize(vmin = min(Acube[:,0]), vmax = max(Acube[:,0])) # 0 is t1, color by t1
        normalize = matplotlib.colors.Normalize(vmin = min(Acube[:,0]), vmax = max(Acube[:,0])) # color by total time

        # plt.scatter([(Acube[x,1] + Acube[x,0]) for x in range(len(Acube[:,2]))],Acube[:,3], s=30, c = Acube[:,0], norm = normalize, cmap = cm) #plotting by total t
        plt.scatter([(Acube[x,2]) for x in range(len(Acube[:,2]))],Acube[:,3], s=30, c = Acube[:,0], norm = normalize, cmap = cm) #plotting by t1

        plt.xlabel('(t1+t2)/t_k') #the kernel timescale is 1/20
        plt.ylabel('y_int/Amax')
    
        # plt.legend()
        plt.title('AUC = 0.05, average val = 25')
        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=normalize, cmap=cm))
        cbar.set_label('t1/t_k', rotation=270, fontsize  = 15)
        plt.savefig(f'{filename}.pdf', format = 'pdf', bbox_inches="tight")





    def test_multiple_sAf3(self): #run this next
        for Am in [100,200,300,400]:
            self.small_Amax_f3_p1(Am,plot='yes')


    def Amax_plot(self,saveword):
        self.kernel = self.initialize_uneven_step(4,0) # I = 0 for now
        aminlist=[]
        Amaxlist =[]
        for A in range(50,400,2):
            Amaxlist.append(A)
            Aminplus, Aminminus = self.small_Amax_f3_p1(A,n=1)
            aminlist.append(Aminplus) #now just plus side #Amin is the average of the two amins, Amin divided by Amax ratio already in fxn
        plt.scatter(Amaxlist,aminlist)
        plt.title('Amin/Amax vs Amax')
        plt.savefig(f'Amax_plot{saveword}.pdf', format = 'pdf', bbox_inches="tight")
        np.save(f'aminlist{saveword}.npy',aminlist)
        np.save(f'Amaxlist{saveword}.npy',Amaxlist)

    def first_nonzero_value(self,x,y,zerolocation):
        step = x[2]-x[1] #needs an evenly spaced A
        startindex=int(np.floor(len(y)/3)) #shoudl set Amin = amax in case it totally fails
        for i,v in enumerate(y[zerolocation:]): #find the start
            if v > 0:
                startindex = i
                break
        right_min_val = startindex * step
      

        frontside=y[:zerolocation]
        frontside.reverse()
        endindex=int(np.floor(len(y)/3))
        for a,b in enumerate(frontside):
            if b < 0:
                endindex=a
                break
        left_min_val = endindex * step

        return right_min_val, left_min_val



    

    def piecewise_Amax(self,x,y,n=9): #pass x y wherte the slope should be 1
        x=np.array(x)
        y=np.array(y)
        my_pwlf = pwlf.PiecewiseLinFit(x, y)
        breaks = my_pwlf.fit(n) #n is number of pieces
        # print(breaks) #breaks has one more entry than slopes
        slopes = my_pwlf.calc_slopes() #also calculates intercepts stored at intercepts
        holdplus = 99
        holdminus = 99
        for i,v in enumerate(slopes):
            if 0.75 < v < 1.25 and (breaks[i+1] - breaks[i]) > 0.15: #finding linear slopes large enough
                if breaks[i+1] > 0 and breaks[i] > 0: #both ends on right
                    if abs(1-v) < abs(1-holdplus): #if best fitting segment
                        holdplus = slopes[i] #new current best sloped segment
                        Aminplus = breaks[i] #make list of y intercepts of pos segs
                if breaks[i+1] < 0 and breaks[i] < 0: #both ends on right
                    if abs(1-v) < abs(1-holdminus): #if best fitting segment
                        holdminus = slopes[i] #new current best sloped segment
                        Aminminus = breaks[i+1] #make list of y intercepts of pos segs
        x_hat = np.linspace(x.min(), x.max(), 100) #100 evenly spaced numbers over the range
        y_hat = my_pwlf.predict(x_hat) #using piecewise to get new graph
        return Aminplus, Aminminus,breaks,slopes,x_hat,y_hat

    def simple_f3_p1(self,filename): #I should probably save this data somewhere
        t1 = 2
        t2 = 10 #slow-fast
        theta = 10
        ds_list = []
        A_list = []
        for j in [0,-50]:
            ds_list = []
            A_list = []
            average=j
            for i in range(1,150,10):
                A_list.append(i+1/10)
                deviation = i/2
                #clamp_value = i+1 + fv
                ds = self.compute_period_weight_update(average - deviation, average + deviation, fast_time = t1, slow_time = t1+t2, nonlinearity_threshold = theta, nonlinearity_style = "linear_threshold")
                ds_list.append(ds)
    
            plt.scatter(A_list, ds_list, marker ="X",label = average)
        plt.title('Synapse Response to Noisy Training',fontsize =20)
        plt.xlabel('Signal Amplitide', fontsize =15)
        plt.ylabel('Synapse Output', fontsize =15)
        #plt.xticks(range(0,151,50)) #make big ticks
        #plt.yticks(range(0,101,50))
        plt.tick_params(axis='both', which='major', labelsize=10, width=4, length=10)

        plt.legend(title = 'Average Value', loc = 'upper left')
        plt.savefig(f'{filename}.pdf', format = 'pdf', bbox_inches="tight")
        plt.show()
        return ds_list
    

    def read_list(self, file):
        with open(file, 'r') as f:
            mylist = [line.rstrip('\n') for line in f]
        return mylist

    def float_list(self, list):
        for i in list:
            i = float(i)
        return list



    def count_linear_values(self, X,Y): 
        count = 0
        for x,y in zip(X,Y):
            if (float(y) > float(x/2)) and (float(y) < float(3*x/2)):
                count += 1
        return(count) #nonzero values  (range of nonzero) # this should be scaled somehow

    def performance_range3(self,x,y,xstep): #assumes x starts at 0
        nonzero_count = 0
        for i,v in enumerate(y):
            if float(v) < float(i/2):
                del v
        for j in y:
            if j != 0:
                nonzero_count += 1
        return (len(y))
            

    def f3_p2(self, tag = 'tag'):
        accuracy_list = [] #I should save these lists to a file
        t1_list = []
        t2_list = []
        tau_list = []
        total_time_list = []
        for k in range (3): #tau
            tau = 0.5 * (k+1)
            #adjust tau and reform kernel
            self.tau = tau
            self.kernel_mesh_size = self.tau/self.dt
            self._initialize_ecoli_kernel()
            print('starting iteration '+ str(k+1) + 'of' + str(3))
            for i in range(10): #t1
                t1 = i+1
                print('part' + str(i))
                for j in range(10): #t2
                
                    t2 = 3*j+1

                    #add to lists
                    tau_list.append(tau)
                    t1_list.append(t1)
                    t2_list.append(t2)
                    total_time_list.append(t1+t2)

                    

                    # do f3p1 and log its r value
                    A_list, ds_list = self.minimal_f3_p1(t1, t2)
                    accuracy_list.append(self.count_linear_values(A_list, ds_list))
        with open(f"32a{tag}.txt", 'w') as f:
            for a in accuracy_list:
                f.write(str(a) + '\n')
        with open(f"32t1{tag}.txt", 'w') as f:
            for a in t1_list:
                f.write(str(a) + '\n')
        with open(f"32t2{tag}.txt", 'w') as f:
            for a in t2_list:
                f.write(str(a) + '\n')
        with open(f"32tau{tag}.txt", 'w') as f:
            for a in tau_list:
                f.write(str(a) + '\n')
        with open(f"32tot{tag}.txt", 'w') as f:
            for a in total_time_list:
                f.write(str(a) + '\n')
        print('done')

    def spread_f3_p2(self, tag = 'tag'):
            accuracy_list = [] #I should save these lists to a file
            t1_list = []
            t2_list = []
            tau_list = []
            total_time_list = []
            for k in range(3): #tau
                tau = (k+2)
                #adjust tau and reform kernel
                self.tau = tau
                self._initialize_ecoli_kernel() #adjust for tau
                print('starting iteration '+ str(k+1) + 'of' + str(10))
                b=0
                while b < 300:
                    b += 1
                
                    t1 = (np.random.randint(1,50))/10
                    t2 = np.random.randint((10*t1),100)/10

                    #add to lists
                    tau_list.append(tau)
                    t1_list.append(t1)
                    t2_list.append(t2)
                    total_time_list.append(t1+t2)

                    

                    # do f3p1 and log its r value
                    A_list, ds_list = self.minimal_f3_p1(t1, t2)
                    accuracy_list.append(self.count_linear_values(A_list, ds_list)/10) #the step between Amplitude values is 4. the intervals are 0.1 so ten 'hits' means a length of 4
            with open(f"32a{tag}.txt", 'w') as f: 
                for a in accuracy_list:
                    f.write(str(a) + '\n')
            with open(f"32t1{tag}.txt", 'w') as f:
                for a in t1_list:
                    f.write(str(a) + '\n')
            with open(f"32t2{tag}.txt", 'w') as f:
                for a in t2_list:
                    f.write(str(a) + '\n')
            with open(f"32tau{tag}.txt", 'w') as f:
                for a in tau_list:
                    f.write(str(a) + '\n')
            with open(f"32tot{tag}.txt", 'w') as f:
                for a in total_time_list:
                    f.write(str(a) + '\n')
            print('done')


        


  #change names before overwriting
    def f3_p2_plots(self, tag): #tag marks the files
        with open(f'32a{tag}.txt', 'r') as f:
            accuracy_list = [float(line.rstrip('\n')) for line in f]
       
        with open(f'32t1{tag}.txt', 'r') as f:
            t1_list = [float(line.rstrip('\n')) for line in f]
       
        with open(f'32t2{tag}.txt', 'r') as f:
            t2_list = [float(line.rstrip('\n')) for line in f]
        
        
        with open(f'32tau{tag}.txt', 'r') as f:
            tau_list = [float(line.rstrip('\n')) for line in f]
           
       
        with open(f'32tot{tag}.txt', 'r') as f:
            total_time_list = [float(line.rstrip('\n')) for line in f]
        # normalize = matplotlib.colors.Normalize(vmin=-1, vmax=1)
        # plt.scatter(total_time_list, accuracy_list, s=5)
        # plt.title('Uncolored Accuracy Plot')
        # plt.xlabel('t1 + t2')
        # plt.ylabel('Range of Linear Performance')
        # plt.xlim([-1,40])
        # plt.ylim([-1,40])
        # plt.savefig(f'uncolored_scatter{tag}.png')
        # plt.close()
        cm = plt.cm.get_cmap('winter')
        #make the colored graphs
        normalize = matplotlib.colors.Normalize(vmin = 0, vmax = max(t1_list))
        plt.scatter(total_time_list, accuracy_list, s=30, c = t1_list, norm = normalize, cmap = cm)
        plt.title('Performance of Kernel With Variable Signals', fontsize = 18)
        plt.xlabel('t1 + t2', fontsize = 15)
        plt.ylabel('Range of Linear Performance', fontsize = 15)
        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=normalize, cmap=cm))
        cbar.set_label('t1', rotation=0, fontsize  = 15)
        plt.xlim([0,10])
        plt.ylim([-0.05,3])
        plt.tick_params(axis='both', which='major', labelsize=10, width=2, length=10)
        plt.savefig(f't1_color_scatter{tag}.pdf', format = 'pdf', bbox_inches="tight")
        plt.close()

        normalize = matplotlib.colors.Normalize(vmin = 0, vmax = max(t2_list))
        plt.scatter(total_time_list, accuracy_list,s=25, c = t2_list, norm = normalize, cmap = cm)
        plt.title('Performance of Kernel With Variable Signals', fontsize =18)
        plt.xlabel('t1 + t2', fontsize = 15)
        plt.ylabel('Range of Linear Performance', fontsize = 15)
        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=normalize, cmap=cm))
        cbar.set_label('t2', rotation=0, fontsize  = 15)
        plt.xlim([0,10])
        plt.ylim([0,3])
        plt.tick_params(axis='both', which='major', labelsize=10, width=4, length=10)
        plt.savefig(f't2_color_scatter{tag}.png')
        plt.close()
        normalize = matplotlib.colors.Normalize(vmin = 0, vmax = max(tau_list))
        plt.scatter(total_time_list, accuracy_list,s=25, c = tau_list, norm = normalize, cmap = cm) #coloring doesn't seem to be working
        plt.title('tau = color')
        plt.xlabel('t1 + t2')
        plt.ylabel('Range of Linear Performance')
        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=normalize, cmap=cm))
        cbar.set_label('tau', rotation=0, fontsize  = 15)
        plt.xlim([0,10])
        plt.ylim([0,10])
        plt.tick_params(axis='both', which='major', labelsize=10, width=2, length=5)
        plt.savefig(f'tau_color_scatter{tag}.png') 
        plt.close()
        print('done')
    
    def vary_f3_p1(self): #this changes values of tau and makes and saves f3_p1 for them
        self.tau = 0.001
        self._initialize_ecoli_kernel()
        self.f3_p1()
        plt.savefig('f3_p1_smalltau')
        plt.close()
        self.tau = 1
        self._initialize_ecoli_kernel()
        self.f3_p1()
        plt.savefig('f3_p1_normaltau')
        plt.close()
        self.tau = 100
        self._initialize_ecoli_kernel()
        self.f3_p1()
        plt.savefig('f3_p1_bigtau')
        plt.close()
        self.tau = 1

    def test_tau_relevence(self):
        mylist = []
        taulist = []
        for i in range(1,14):
            taulist.append(np.log10(self.tau))
            mylist.append(self.compute_period_weight_update(0,10,(2*i)/10,(10*i)/10,5/((2*i)/10)))
            print('iteration')
        error_ratio = []
        for item in mylist:
            error_ratio.append(abs((10-item)/10))
        self.tau =1 #restore tau
        plt.scatter([(2*i)/10 for i in range(1,14)], error_ratio)
        plt.title('Relative Timescales vs Error Rate')
        plt.xlabel('sawtooth timescale')
        plt.ylabel('ratio of error')
        plt.show()
        return error_ratio


    def initialize_uneven_step(self,absarea = 4,area=0):
        a = absarea + area
        b = absarea - area
        kernel = np.ones(self.kernel_mesh_size)
        kernel[len(kernel)//2:] = -b 
        kernel[:len(kernel)//2] = a 
        self.kernel = kernel
        return kernel

    #def measure_scatter():

    def locate_bounds():

        return x1, x2

    def truncatedrsquare(self,x,y, start, stop):
        truncatedA = x[start:stop]
        truncatedds = y[start:stop]
        r = self.r_value(truncatedA,truncatedds)
        rsquared = r**2

    
    def failure_stats(self,x,A,zerolocation):
        step = A[2]-A[1]
        startindex=zerolocation
        for i,v in enumerate(x[zerolocation:]): #find the start
            if v == 0:
                startindex = i
            if v > 0 and startindex == i-1:
                break
        storage = 0
        endindex=None
        for a,b in enumerate(x[zerolocation:]): #find the end
            if b < storage:
                endindex=a
                break
            storage = b
        dist = (endindex-startindex)*step

        s,intercept,r,p,sig = self.r_value(A[startindex:endindex],x[startindex:endindex])
        intercept = np.abs(intercept)


        frontside=x[:startindex]*-1
        frontside.reverse()
        storage = 0
        for a,b in enumerate(frontside):
            if b < storage:
                endindex=a
                break
            storage = b
        dist2 = (endindex-startindex)*step
        totdist = dist + dist2
        asym = np.abs((dist-dist2)/totdist)
        return totdist,intercept,asym #total range of linear response, that which is positive, negative, abs(intercept) of positive linear sigment, asymmetry ratio
    
    
    def singlet(self):
        self.initialize_uneven_step(4,0.05)
        Alist = []
        dslist = []
        av = 10
        for A in range(-100,100,20):
            Alist.append(A)
            dslist.append(self.compute_period_weight_update(av-(A/2),av+(A/2),2,5,10))

        plt.plot(Alist,dslist,lw=4,c='black')
        plt.ylabel('Synapse Response')
        plt.xlabel('Signal Amplitude')
        plt.savefig('singlet2.pdf', format = 'pdf', bbox_inches="tight")
        plt.show()
        return Alist,dslist

    def workhorse_singlet(self,av,t1,t2):
        self.initialize_uneven_step(4,0.05)
        Alist = []
        dslist = []
        for A in range(-500,500,10):
            Alist.append(A)
            dslist.append(self.compute_period_weight_update(av-(A/2),av+(A/2),t1,t1+t2,5))
        return Alist,dslist

    def megaparams(self):
        megacube = np.zeros((3*10*10,7))
        counter = 0
        for av in [50]: #this is all for av = 50 
            i=0
            while i < 300:
                if i%10 == 0:
                    print('iteration' + str(i))
                i+=1
                t1 = np.random.randint(1,200)/10 #generate mostly random values for times
                t2 = np.random.randint(int(30*t1),1000)/10
                Alist,dslist = self.workhorsetimelet(t1,t2)
                totdist,intercept,asym = self.piecewise_failure_stats(Alist,dslist,10)
                print(f'{totdist}{intercept}{asym}')
                megacube[counter] = [av,t1+t2,t1,t2,totdist,intercept,asym] #replace row with relevant data
                counter+= 1
        np.save('megacube.npy',megacube)
    
    def separate_cubes(self):
        Acube = np.zeros((300,100))
        dscube = np.zeros((300,100))
        breakscube = np.zeros((300,11))
        slopescube = np.zeros((300,10))
        xhcube = np.zeros((300,100))
        yhcube = np.zeros((300,100))
        errorcube = np.zeros((300,7))
        counter = 0
        for av in [50]: #this is all for av = 50 
            i=0
            while i < 300:
                if i%10 == 0:
                    print('iteration' + str(i))
                i+=1
                t1 = np.random.randint(1,200)/10 #generate mostly random values for times
                t2 = np.random.randint(int(30*t1),1000)/10
                Alist,dslist = self.workhorsetimelet(t1,t2)
                totdist,intercept,asym,breaks,slopes,intercepts,x_hat,y_hat = self.piecewise_failure_stats(Alist,dslist,10)
                print(f'{totdist}{intercept}{asym}')
                errorcube[counter] = [av,t1+t2,t1,t2,totdist,intercept,asym] #replace row with relevant data
                Acube[counter] = Alist
                dscube[counter] = dslist
                breakscube[counter] = breaks
                slopescube[counter] = slopes
                xhcube[counter] = x_hat
                yhcube[counter] = y_hat
                counter+= 1
        np.save('Acube.npy',Acube)
        np.save('dscube.npy',dscube)
        np.save('breakscube.npy',breakscube)
        np.save('slopescube.npy',slopescube)
        np.save('xhcube.npy',xhcube)
        np.save('yhcube.npy',yhcube)
        np.save('errorcube.npy',errorcube)

    def justkernels(self):
        Acube = np.zeros((300,90))
        dscube = np.zeros((300,90))
        counter = 0
        for av in [50]: #this is all for av = 50 
            i=0
            while i < 300:
                if i%10 == 0:
                    print('iteration' + str(i))
                i+=1
                t1 = np.random.randint(1,200)/10 #generate mostly random values for times
                t2 = np.random.randint(int(30*t1),800)/10
                Alist,dslist = self.workhorsetimelet(t1,t2)
                Acube[counter] = Alist #replace row with relevant data
                dscube[counter] = dslist
                counter+= 1
        np.save('Acube2.npy',Acube)
        np.save('dscube2.npy',dscube)

    
    def make_megascatters(): #predicated on 3 average values w/ 100 trials each? wronggg
        errorcube=np.load('errorcube_c.npy')
        timecube=np.load('timecube_c.npy')

        cm = plt.cm.get_cmap('winter')



        for avindex in [0]:#for avindex in (0,100,200)
            for failurestat in [4,5,6]:
                failname = ['a','a','a','a','Total Linear Range','Y Intercept Magnitude', 'Asymmetry Ratio']
                #make the colored graphs
                normalize = matplotlib.colors.Normalize(vmin = 0, vmax = max(megacube[avindex:avindex+300,2]))
                plt.scatter(megacube[avindex:avindex+300, 1],megacube[avindex:avindex+300,failurestat] , s=30, c = megacube[avindex:avindex+300,2], norm = normalize, cmap = cm)#color by t1
                plt.title('Kernel Failure vs Timescale', fontsize = 18)
                plt.xlabel('Signal Timescale', fontsize = 15)
                plt.ylabel(f'{failname[failurestat]}', fontsize = 15)
                cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=normalize, cmap=cm))
                cbar.set_label('t1', rotation=0, fontsize  = 15)
                plt.tick_params(axis='both', which='major', labelsize=10, width=2, length=10)
                plt.savefig(f'f5p2{failurestat}_{(avindex/100)+1}.pdf', format = 'pdf', bbox_inches="tight") #will this go where it needs to on midway?
                plt.close()

    def make_scatter2(self): #predicated on 3 average values w/ 100 trials each? wronggg
        errorcube=np.load('shearederrorcube.npy')
        timecube=np.load('shearedtimecube.npy')

        cm = plt.cm.get_cmap('winter')



        #make the colored graphs
        normalize = matplotlib.colors.Normalize(vmin = 0, vmax = max(timecube[:,2]))
        plt.scatter(timecube[:,0],errorcube[:,1] , s=30, c = timecube[:,2], norm = normalize, cmap = cm)#color by t1+t2, plot by t1
        plt.title('Kernel Failure vs Timescale', fontsize = 18)
        plt.xlabel('t1', fontsize = 15)
        plt.ylabel(f'Y Intercept', fontsize = 15)
        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=normalize, cmap=cm))
        cbar.set_label('t1+t2', rotation=0, fontsize  = 15)
        plt.tick_params(axis='both', which='major', labelsize=10, width=2, length=10)
        plt.savefig(f'newt1plot.pdf', format = 'pdf', bbox_inches="tight") #will this go where it needs to on midway?
        plt.close()

    def compare_failure_metrics(self):
        x = [*range(300)]
        megacube = np.load('megacube.npy')
        failname = ['a','a','a','a','Total Linear Range','Y Intercept Magnitude', 'Asymmetry Ratio']
        for i in (4,5,6):
            plt.plot(x,megacube[:,i]/(megacube[:,i].max()), label = failname[i]) #plotting normalized errors
            plt.xlabel('Sample Case')
            plt.ylabel('Failure Modes')
        plt.legend(loc = 'upper left')
        plt.savefig(f'Failure_Stats.pdf', format = 'pdf', bbox_inches="tight")
        plt.close()

    def piecewise_failure_stats(self,x,y,n=10):
        x=np.array(x)
        y=np.array(y)
        my_pwlf = pwlf.PiecewiseLinFit(x, y)
        breaks = my_pwlf.fit(n) #n is number of pieces
        # print(breaks) #breaks has one more entry than slopes
        slopes = my_pwlf.calc_slopes() #also calculates intercepts stored at intercepts
        intercepts = my_pwlf.intercepts
        intercept=12 #diagnostic
        poslinear_length=0
        neglinear_length =0
        hold = 99
        for i,v in enumerate(slopes):
            if 0.75 < v < 1.25 and (breaks[i+1] - breaks[i]) > .1: #finding linear slopes large enough
                if breaks[i+1] > 0: 
                    if abs(1-v) < abs(1-hold): #if best fitting segment
                        hold = slopes[i] #new current best sloped segment
                        intercept = intercepts[i] #make list of y intercepts of pos segs
                    poslinear_length += (breaks[i+1] - breaks[i]) #add that segment to linear
                else:
                    neglinear_length += (breaks[i+1] - breaks[i]) #add that segment to linear (does include 0
        
        linlength = poslinear_length + neglinear_length + 0.1
        lenratio = poslinear_length/linlength
        x_hat = np.linspace(x.min(), x.max(), 100) #100 evenly spaced numbers over the range
        y_hat = my_pwlf.predict(x_hat) #using piecewise to get new graph
        return linlength,intercept,lenratio,breaks,slopes,intercepts,x_hat,y_hat

    def simplesinglet(self): #making a simple explanatory kernel
            self.kernel = self.initialize_uneven_step(4,0.1)
            t1 = 4
            t2 = 20 #slow-fast
            theta = 10
            ds_list = []
            A_list = []
            cm = plt.cm.get_cmap('coolwarm')
            for j in [50]: #for j in [-100,-50,0,50,100]:
                ds_list = []
                A_list = []
                average=j
                for i in range(-10,250,5):
                    A_list.append(i+1/10)
                    deviation = i/2
                    #clamp_value = i+1 + fv
                    ds = self.compute_period_weight_update(average - deviation, average + deviation, fast_time = t1, slow_time = t1+t2, nonlinearity_threshold = theta, nonlinearity_style = "linear_threshold")
                    ds_list.append(ds)
                color=cm((j+100)/200)
                plt.plot(A_list, ds_list,c=color, lw = 4, label = average)
            plt.title('Synapse Response to Noisy Training',fontsize =20)
            plt.xlabel('Signal Amplitide', fontsize =15)
            plt.ylabel('Synapse Output', fontsize =15)
            #plt.xticks(range(0,151,50)) #make big ticks
            #plt.yticks(range(0,101,50))
            plt.tick_params(axis='both', which='major', labelsize=10, width=4, length=10)
            #plt.xlim(-10,250)
            #plt.ylim(-10,110)
            # plt.legend(title = 'Average Value', loc = 'upper left')
            plt.savefig('simple.pdf', format = 'pdf', bbox_inches="tight")
            plt.show()
            return ds_list
        
    def simpledoublet(self): #making a simple explanatory kernel
            self.kernel = self.initialize_uneven_step(4,0.05)
            t1 = 2
            t2 = 10 #slow-fast
            theta = 10
            ds_list = []
            A_list = []
            cm = plt.cm.get_cmap('coolwarm')
            for j in [-50,50]: #for j in [-100,-50,0,50,100]:
                ds_list = []
                A_list = []
                average=j
                for i in range(-120,120,5):
                    A_list.append(i+1/10)
                    deviation = i/2
                    #clamp_value = i+1 + fv
                    ds = self.compute_period_weight_update(average - deviation, average + deviation, fast_time = t1, slow_time = t1+t2, nonlinearity_threshold = theta, nonlinearity_style = "linear_threshold")
                    ds_list.append(ds)
                color=cm((j+100)/200)
                plt.plot(A_list, ds_list,c=color, lw = 6, label = average)
            plt.title('Shifted Base Value',fontsize =20)
            plt.xlabel('Signal Amplitide', fontsize =15)
            plt.ylabel('Synapse Output', fontsize =15)
            #plt.xticks(range(0,151,50)) #make big ticks
            #plt.yticks(range(0,101,50))
            plt.tick_params(axis='both', which='major', labelsize=10, width=4, length=10)
            #plt.xlim(-10,120)
            #plt.ylim(-10,110)
            plt.legend(title = 'Average Value', loc = 'upper left')
            plt.savefig('doublet.pdf', format = 'pdf', bbox_inches="tight")
            plt.show()
            return ds_list
    
    def workhorsetimelet(self,t1,t2): #making a simple explanatory kernel
            self.kernel = self.initialize_uneven_step(4,0.1)
            theta = 10
            ds_list = []
            A_list = []
            average = 50
            ds_list = []
            A_list = []
            for i in range(-400,500,10):
                A_list.append(i+1/10)
                deviation = i/2
                #clamp_value = i+1 + fv
                ds = self.compute_period_weight_update(average - deviation, average + deviation, fast_time = t1, slow_time = t1+t2, nonlinearity_threshold = theta, nonlinearity_style = "linear_threshold")
                ds_list.append(ds)
            return A_list, ds_list

    def test_small_t1(self): #nit giving expected results
        dslist =[]
        tlist = []
        for i in range(1,200):
            t1 = i/1000
            tlist.append(t1*20)
            dslist.append(self.compute_period_weight_update(0, 100, t1, slow_time = 5, nonlinearity_threshold = 50, nonlinearity_style = "linear_threshold", downswing ='yes')/100)
        plt.plot(tlist,dslist)
        plt.ylabel('ds/A')
        plt.xlabel('t1/t_k')
        plt.savefig('smallt1test.pdf', format = 'pdf', bbox_inches="tight")
        plt.show()

    # def 
    # self.kernel = self._initialize_uneven_ecoli_kernel(I)

#plot total time vs dissipation for some fixed params (should just be a line by measurement = A + (total time)(average value)(AUC))
#because measured derivative = derivatiove + AUC

#testing if it works
#---------------------
Sim = ActiveFlowNetworks()


#Sim.task_id=os.environ.get('SLURM_ARRAY_TASK_ID')     #these get slurm sbatch variables in array job
#Sim.max_task_id = os.environ.get('SLURM_ARRAY_TASK_MAX')
# print(os.environ.get('SLURM_ARRAY_TASK_ID'))

Sim.kernel = Sim._initialize_ecoli_kernel()
print('starting')

Sim.midway_expanded_Amax_f3_p2_cube(0.05,25) #make data for nonzero AUC ad base value
 #will initialize kernel and run midway version, using taskID to run part of code

#Sim.f3_p2()


#call the errormap
#x_list, y_list, heatlist = Sim.ratio_errormap(5,15)
#Sim.plot_errormap(x_list, y_list, heatlist, 5,20,5)

#plt.plot(Sim.parameter_scatterplot(free_value = 0)) #show an example sawtooth range vs AUC graph
#plt.show()


# Sim.variation_of_parameters(0) #iterate over timescales, nonlinearity threshold and plot all r_values between sawtooth range and AUC haphazardly

# Sim.kernel = Sim._initialize_sin_kernel()



# A list of tasks/functions you might need to build in order to carry out the calculation:
# 1. a function to construct the external signal, (DONE)
# given a fast rate, a slow rate, a free value, and a clamped value.
# 2. a function which calls compute_period_weight_update for multiple (DONE)
# free and clamped values, to construct the scatterplot we discussed. 
# 3. a function to compute the r-values of the scatter of free-clamped vs the integral. (DONE)
# 4. a function which systematically iterates over the 3-dim space of rates (DONE)
# we discussed, and calls the r-value function for each triplet.

# There are many ways to go about this, so if you think of something you want to try,
# go for it!

# what may be useful is to write an x^3 function which will capture eq prop worse but be robust to shifting thresholds)