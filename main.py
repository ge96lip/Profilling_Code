import time
from functools import wraps
from timeit import default_timer as timer
import numpy as np
import cProfile
import subprocess
NAME = "carlotta"
"""Julia set generator without optional PIL-based image drawing"""

# area of complex space to investigate
x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
c_real, c_imag = -0.62772, -.42193
# decorator to time
def timefn(fn):
  @wraps(fn)
  def measure_time(*args, **kwargs):
    t1 = time.time()
    result = fn(*args, **kwargs)
    t2 = time.time()
    print(f"@timefn: {fn.__name__} took {t2 - t1} seconds")
    return result
  return measure_time

def calc_pure_python(desired_width, max_iterations):
  """Create a list of complex coordinates (zs) and complex parameters (cs),
  build Julia set"""
  x_step = (x2 - x1) / desired_width
  y_step = (y1 - y2) / desired_width
  x = []
  y = []
  ycoord = y2
  while ycoord > y1:
    y.append(ycoord)
    ycoord += y_step
  xcoord = x1
  while xcoord < x2:
    x.append(xcoord)
    xcoord += x_step
  # build a list of coordinates and the initial condition for each cell.
  # Note that our initial condition is a constant and could easily be removed,
  # we use it to simulate a real-world scenario with several inputs to our
  # function
  zs = []
  cs = []
  for ycoord in y:
    for xcoord in x:
      zs.append(complex(xcoord, ycoord))
      cs.append(complex(c_real, c_imag))
  print("Length of x:", len(x))
  print("Total elements:", len(zs))
  start_time = time.time()
  output = calculate_z_serial_purepython(max_iterations, zs, cs)
  end_time = time.time()
  secs = end_time - start_time
  print(calculate_z_serial_purepython.__name__ + " took", secs, "seconds")
  # This sum is expected for a 1000^2 grid with 300 iterations
  # It ensures that our code evolves exactly as we'd intended
  assert sum(output) == 33219980
  return [zs, cs]


def calculate_z_serial_purepython(maxiter, zs, cs):
  """Calculate output list using Julia update rule"""
  output = [0] * len(zs)
  for i in range(len(zs)):
    n = 0
    z = zs[i]
    c = cs[i]
    while abs(z) < 2 and n < maxiter:
      z = z * z + c
      n += 1
    output[i] = n
  return output

# Task 1.1: Calculate the Clock Granularity of different Python Timers
outfile = None
def checktick_0():
    """
    time.time(): precision is approximately ± 1 microsecond
    """
    M = 200
    timesfound = np.empty((M,))
    for i in range(M):
        t1 =  time.time() # get timestamp from timer
        t2 = time.time() # get timestamp from timer
        while (t2 - t1) < 1e-16: # if zero then we are below clock granularity, retake timing
            t2 = time.time() # get timestamp from timer
        t1 = t2 # this is outside the loop
        timesfound[i] = t1 # record the time stamp
    minDelta = 1000000
    Delta = np.diff(timesfound) # it should be cast to int only when needed
    minDelta = Delta.min()
    return minDelta

def checktick_1():
    """
    timeit: higher resolution, lower clock granularity
    """
    M = 200
    timesfound = np.empty((M,))
    for i in range(M):
        t1 =  timer() # get timestamp from timer
        t2 = timer() # get timestamp from timer
        while (t2 - t1) < 1e-16: # if zero then we are below clock granularity, retake timing
            t2 = timer() # get timestamp from timer
        t1 = t2 # this is outside the loop
        timesfound[i] = t1 # record the time stamp
    minDelta = 1000000
    Delta = np.diff(timesfound) # it should be cast to int only when needed
    minDelta = Delta.min()
    return minDelta

def checktick_2():
    """
    higher resolution in nanoseconds
    """
    M = 200
    timesfound = np.empty((M,))
    for i in range(M):
        t1 =  time.time_ns() # get timestamp from timer
        t2 = time.time_ns() # get timestamp from timer
        while (t2 - t1) < 1e-16: # if zero then we are below clock granularity, retake timing
            t2 = time.time_ns() # get timestamp from timer
        t1 = t2 # this is outside the loop
        timesfound[i] = t1 # record the time stamp
    minDelta = 1000000
    Delta = np.diff(timesfound) # it should be cast to int only when needed
    minDelta = Delta.min()
    return minDelta


def task_1_1():
  print("----------Exercise 1: Profiling the Julia Set Code----------\n", file=outfile)
  print("-------------------------Task 1.1-------------------------", file=outfile)
  print("time.time() : ", checktick_0(), file=outfile)
  print("timeit : ", checktick_1(), file=outfile)
  print("time.time_ns() : ", checktick_2()/1000000000, "\n", file=outfile)

# Task 1.2
# decorator developed for task 1.2
def timefn_1_2(fn):
  @wraps(fn)
  def measure_time(*args, **kwargs):
    times = np.zeros(3, dtype=float)
    for i in range(3):
      t1 = timer()
      result = fn(*args, **kwargs)
      t2 = timer()
      times[i] = t2-t1
    print(f"@timefn_1_2: {fn.__name__} took {np.average(times)} seconds on average, with standard deviation {np.std(times)}", file=outfile)
    return result
  return measure_time

params_zfunc = []

def task_1_2():
  print("-------------------------Task 1.2-------------------------", file=outfile)
  print("Profiling calc_pure_python:", file=outfile)
  test1 = timefn_1_2(calc_pure_python)
  params2 = test1(desired_width=1000, max_iterations=300)
  print("\n Profiling calculate_z_serial_purepython", file=outfile)
  test2 = timefn_1_2(calculate_z_serial_purepython)
  test2(300, params2[0], params2[1])
  print("\n")
  return params2

# Task 1.3
def func_to_run():
    return calculate_z_serial_purepython(300, params_zfunc[0], params_zfunc[1])

def task_1_3():
  print("-------------------------Task 1.3-------------------------", file=outfile)
  cProfile.run('calc_pure_python(desired_width=1000, max_iterations=300)', filename=f'profile0_{NAME}.stats')
  runnable = 'func_to_run()'
  cProfile.run(runnable, filename=f'profile1_{NAME}.stats')
  print("snakeviz calc_pure_python", file=outfile)
  out1 = subprocess.check_output(["python", "-m", "snakeviz", f"profile0_{NAME}.stats", "--server", ">", f"profile0_{NAME}.txt"], shell=True, timeout=10)
  print(out1)
  print("snakeviz calculate_z_serial_purepython", file=outfile)
  out2 = subprocess.check_output(["python", "-m", "snakeviz", f"profile1_{NAME}.stats", "--server", ">", f"profile1_{NAME}.txt"], shell=True, timeout=10)
  print(out2)
  
if __name__ == "__main__":
    outfile = open(f"out_{NAME}.txt", "w")
    print("-------------------------Task 1.1-------------------------")
    task_1_1()
    print("-------------------------Task 1.2-------------------------")
    params_zfunc = task_1_2()
    print("-------------------------Task 1.3-------------------------")
    task_1_3()
    outfile.close()