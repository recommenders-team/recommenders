# Original code: https://raw.githubusercontent.com/miguelgfierro/codebase/master/python/system/notebook_memory_management.py
#
# Profile memory usage envelope of IPython commands and report interactively.
# Usage (inside a python notebook):
#   from notebook_memory_management import start_watching_memory, stop_watching_memory
# To start profile:
#   start_watching_memory()
# To stop profile:
#   stop_watching_memory()
#
# Based on: https://github.com/ianozsvald/ipython_memory_usage
#

from __future__ import division  # 1/2 == 0.5, as in Py3
from __future__ import absolute_import  # avoid hiding global modules with locals
from __future__ import print_function  # force use of print("hello")
from __future__ import (
    unicode_literals
)  # force unadorned strings "" to be Unicode without prepending u""
import time
import memory_profiler
from IPython import get_ipython
import psutil
import warnings


# keep a global accounting for the last known memory usage
# which is the reference point for the memory delta calculation
previous_call_memory_usage = memory_profiler.memory_usage()[0]
t1 = time.time()  # will be set to current time later
keep_watching = True
watching_memory = True
try:
    input_cells = get_ipython().user_ns["In"]
except:
    warnings.warn("Not running on notebook")


def start_watching_memory():
    """Register memory profiling tools to IPython instance."""
    global watching_memory
    watching_memory = True
    ip = get_ipython()
    ip.events.register("post_run_cell", watch_memory)
    ip.events.register("pre_run_cell", pre_run_cell)


def stop_watching_memory():
    """Unregister memory profiling tools from IPython instance."""
    global watching_memory
    watching_memory = False
    ip = get_ipython()
    try:
        ip.events.unregister("post_run_cell", watch_memory)
    except ValueError:
        print("ERROR: problem when unregistering")
        pass
    try:
        ip.events.unregister("pre_run_cell", pre_run_cell)
    except ValueError:
        print("ERROR: problem when unregistering")
        pass


def watch_memory():
    # bring in the global memory usage value from the previous iteration
    global previous_call_memory_usage, keep_watching, watching_memory, input_cells
    new_memory_usage = memory_profiler.memory_usage()[0]
    memory_delta = new_memory_usage - previous_call_memory_usage
    keep_watching = False
    total_memory = psutil.virtual_memory()[0] / 1024 / 1024  # in Mb
    # calculate time delta using global t1 (from the pre-run event) and current time
    time_delta_secs = time.time() - t1
    num_commands = len(input_cells) - 1
    cmd = "In [{}]".format(num_commands)
    # convert the results into a pretty string
    output_template = (
        "{cmd} used {memory_delta:0.4f} Mb RAM in "
        "{time_delta:0.2f}s, total RAM usage "
        "{memory_usage:0.2f} Mb, total RAM "
        "memory {total_memory:0.2f} Mb"
    )
    output = output_template.format(
        time_delta=time_delta_secs,
        cmd=cmd,
        memory_delta=memory_delta,
        memory_usage=new_memory_usage,
        total_memory=total_memory,
    )
    if watching_memory:
        print(str(output))
    previous_call_memory_usage = new_memory_usage


def pre_run_cell():
    """Capture current time before we execute the current command"""
    global t1
    t1 = time.time()

