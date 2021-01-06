
"""
Description:
    This is a simple script to demonstrate Scipy's curve_fit module using a few common mathematical functions.
============================================================================================
License_info:
    LCD-Composer is subject to the terms of the MIT license.
"""

__author__ = 'Sean M Cascarina'
__copyright__ = 'Copyright 2021'
__credits__ = ['Sean M Cascarina']
__license__ = 'MIT'
__version__ = '1.0'
__maintainer__ = 'Sean M Cascarina'
__email__ = 'Sean.Cascarina@colostate.edu'


# IMPORTS
from scipy.optimize import curve_fit
import random
import matplotlib.pyplot as plt
import math
import numpy as np

def main():

    # COMMENT OR UNCOMMENT THESE LINES TO RUN DIFFERENT SIMULATIONS
    print('\n')
    run_linear_fit()
    run_gaussian_fit()
    run_quadratic_fit()
    run_cubic_fit()
    run_sine_fit()
    print('\n')


def lin_objective(x, m, b):
    """Linear objective function"""
    return m*x + b
    
    
def run_linear_fit(slope=2, xnoise=0.2, ynoise=0.4):
    """Run linear fit simulation.
        By default, the slope in the simulated data is == 2."""
    
    # SIMPLE LINEAR EXAMPLE=============================================================
    xdata = [x+random.uniform(-xnoise, xnoise) for x in range(1, 21)]
    ydata = [lin_objective(x, slope, random.uniform(-ynoise, ynoise)) for x in range(1, 21)]
    
    popt, pcov = curve_fit(lin_objective, xdata, ydata)
    m, b = popt

    print('Linear estimated parameters:', 'Slope(m)='+str(m), 'Intercept(b)='+str(b))
    
    plt.scatter(xdata, ydata)
    plt.plot([x for x in range(1, 21)], [lin_objective(x, m, b) for x in range(1, 21)], color='#d62728')
    plt.show()
    #===================================================================================
    
    
def gauss_objective(x, amplitude, sigma, mu):
    """Gaussian distribution objective function.
        This differs from many normal distribution equations online but the amplitude is required to
        stretch the Gaussian upward to match your specific data size.
        
        This equation was adapted from: https://stackoverflow.com/questions/19206332/gaussian-fit-for-python,
        but also matches the first equation on the Gaussian function Wikipedia page: https://en.wikipedia.org/wiki/Gaussian_function,
        and can be found elsewhere in this form as well."""
    
    # return amplitude * (1 / (math.sqrt(2*math.pi*sigma**2))) * math.e**(-1 * (x-mu)**2 / 2*sigma**2)
    # return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return amplitude * math.e**(-(x - mu)**2 / (2 * sigma**2))
    
    
def run_gaussian_fit(sigma=1, mu=0, noise=0.1):
    """Run Gaussian fit simulation"""
    
    # GENERATE A SAMPLE OF NORMALLY DISTRIBUTED DATA
    data = np.random.normal(mu, sigma, 1000)
    data = [x+random.uniform(-noise, noise) for x in data]  # ADD NOISE TO THE DATA
    
    # PLOT HISTOGRAM AND STORE BIN LOCATIONS AND CORRESPONDING FREQUENCIES IN xvals AND yvals, RESPECTIVELY
    yvals, xvals, patches = plt.hist(data, bins=30)
    
    # SHIFT xvals BY HALF OF THE bin_width TO ACCOUNT FOR THE FACT THAT plt.hist() RETURNS AN ARRAY WITH THE LEFT BIN EDGES AND THE FINAL ELEMENT IN THE LIST BEING THE FINAL RIGHT BIN EDGE
    # THIS ADJUSTMENT MAKES THE CURVE FITTING PARAMETERS MORE ACCURATE/CENTERED, BUT ALSO SEEMS TO CONSISTENTLY RESULT IN A NEGATIVE STANDARD DEVIATION ESTIMATE AND ALL VALUES END UP BEING IN SCIENTIFIC NOTATION AFTER THIS ADJUSTMENT
    bin_width = xvals[-1] - xvals[-2]
    # print(bin_width)
    xvals = [x+bin_width/2 for x in xvals[:-1]]
    
    # CALCULATE amplitude AS MAXIMUM OF yvals...THIS IS USED TO STRETCH THE GAUSSIAN CURVE TO ACTUALLY FIT OUR DATA SIZE
    amplitude = max(yvals)
    
    # RUN CURVE FITTING WITH THE xvals AND yvals DETERMINED FROM THE HISTOGRAM
    popt, pcov = curve_fit(gauss_objective, xvals, yvals)
    a, s, m = popt

    print('Gaussian estimated parameters:', 'Amplitude='+str(a), 'Sigma='+str(s), 'Mu='+str(m))

    # CALCULATE VALUES TO PLOT THE GAUSSIAN FIT USING THE PARAMETERS DETERMINED FROM curve_fit()
    fit_data = [gauss_objective(x, max(yvals), s, m) for x in xvals]
    plt.plot(xvals, fit_data, color='#d62728')
    plt.show()
    
    
def quadratic_objective(x, a, b, c):
    """Quadratic objective function."""
    return a*x**2 + b*x + c
    
    
def run_quadratic_fit(a=1, b=1, c=1, noise=5, data_range=100):
    """Run quadratic fit simulation."""
    
    # SAMPLE DATA USING OBJECTIVE FUNCTION
    xvals = np.linspace(-data_range, data_range, num=100)
    yvals = [quadratic_objective(x, a, b, c) for x in xvals]
    
    # ADD NOISE TO DATA
    xdata = [x+random.uniform(-noise, noise) for x in xvals]
    # ydata = [y+random.uniform(-noise, noise) for y in yvals]
    ydata = [y+quadratic_objective(random.uniform(-noise, noise), a, b, c)-c for y in yvals]  # USES THE QUADRATIC OBJECTIVE FUNCTION TO MAKE THE NOISE PROPORTIONAL IN MAGNITUDE ON THE Y-AXIS
    
    # RUN CURVE FITTING USING SIMULATED DATA
    popt, pcov = curve_fit(quadratic_objective, xdata, ydata)
    
    # UNPACK PARAMETER ESTIMATES
    est_a, est_b, est_c = popt
    print('Quadratic estimated parameters:', 'a='+str(est_a), 'b='+str(est_b), 'c='+str(est_c))
    
    # PLOT SIMULATED DATA IN SCATTER PLOT (BLUE), AND CURVE FIT FUNCTION AS A LINE (RED)
    plt.scatter(xdata, ydata)
    plt.plot([x for x in range(-data_range, data_range)], [quadratic_objective(x, est_a, est_b, est_c) for x in range(-data_range, data_range)], color='#d62728')
    plt.show()
    
    
def cubic_objective(x, a, b, c, d):
    """Cubic objective function."""
    return a*x**3 + b*x**2 + c*x + d
    
    
def run_cubic_fit(a=1, b=1, c=1, d=1, noise=5, data_range=100):
    """Run cubic fit simulation."""
    
    # SAMPLE DATA USING OBJECTIVE FUNCTION
    xvals = np.linspace(-data_range, data_range, num=100)
    yvals = [cubic_objective(x, a, b, c, d) for x in xvals]
    
    # ADD NOISE TO DATA
    xdata = [x+random.uniform(-noise, noise) for x in xvals]
    # ydata = [y+random.uniform(-noise, noise) for y in yvals]
    ydata = [y+cubic_objective(random.uniform(-noise, noise), a, b, c, d)-d for y in yvals]  # USES THE CUBIC OBJECTIVE FUNCTION TO MAKE THE NOISE PROPORTIONAL IN MAGNITUDE ON THE Y-AXIS
    
    # RUN CURVE FITTING USING SIMULATED DATA
    popt, pcov = curve_fit(cubic_objective, xdata, ydata)
    
    # UNPACK PARAMETER ESTIMATES
    est_a, est_b, est_c, est_d = popt
    print('Cubic estimated parameters:', 'a='+str(est_a), 'b='+str(est_b), 'c='+str(est_c), 'd='+str(est_d))
    
    # PLOT SIMULATED DATA IN SCATTER PLOT (BLUE), AND CURVE FIT FUNCTION AS A LINE (RED)
    plt.scatter(xdata, ydata)
    plt.plot([x for x in range(-data_range, data_range)], [cubic_objective(x, est_a, est_b, est_c, est_d) for x in range(-data_range, data_range)], color='#d62728')
    plt.show()
    
    
def sine_objective(x, a, b):
    """Sine objective function."""
    return a*np.sin(x) + b
    
    
def run_sine_fit(a=2, b=5, noise=0.1, data_range=10):
    """Run sine fit simulation."""
    
    # SAMPLE DATA USING OBJECTIVE FUNCTION
    xvals = np.linspace(-data_range, data_range, num=100)
    yvals = [sine_objective(x, a, b) for x in xvals]
    
    # ADD NOISE TO DATA
    xdata = [x+random.uniform(-noise, noise) for x in xvals]
    # ydata = [y+random.uniform(-noise, noise) for y in yvals]
    ydata = [y+sine_objective(random.uniform(-noise, noise), a, b)-b for y in yvals]  # USES THE SINE OBJECTIVE FUNCTION TO MAKE THE NOISE PROPORTIONAL IN MAGNITUDE ON THE Y-AXIS. SUBTRACTING THE INTERCEPT (b) IS NECESSARY SO THAT THE INTERCEPT ISN'T ADDED TO EACH Y VALUE TWICE.
    
    # RUN CURVE FITTING USING SIMULATED DATA
    popt, pcov = curve_fit(sine_objective, xdata, ydata)
    
    # UNPACK PARAMETER ESTIMATES
    est_a, est_b = popt
    print('Sine function estimated parameters', 'a='+str(est_a), 'b='+str(est_b))
    
    # PLOT SIMULATED DATA IN SCATTER PLOT (BLUE), AND CURVE FIT FUNCTION AS A LINE (RED)
    plt.scatter(xdata, ydata)
    plt.plot(xvals, [sine_objective(x, est_a, est_b) for x in xvals], color='#d62728')
    plt.show()
    

if __name__ == '__main__':
    main()