import numpy as np

def convert_to_dataset(trace_list):
    trace_data_list = []
    for trace in trace_list:
        trace_data = {}
        for varname in trace.varnames:
            trace_data[varname] = trace[varname]
        trace_data_list.append(trace_data)
    return trace_data_list