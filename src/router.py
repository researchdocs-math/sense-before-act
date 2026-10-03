"""Execute a compiled sensing table against a fixed binary sensor snapshot."""
import time

def route(spec,policy,z,budget):
    h=0;b=int(budget);trace=[];start=time.perf_counter_ns()
    while True:
        a=int(policy[h,b])
        if a<0:return a==-2,trace,(time.perf_counter_ns()-start)/1000
        trace.append({'sensor':spec['features'][a],'source':int(spec['groups'][a]),'report':int(z[a]),'cost':int(spec['costs'][a])})
        h+=(1+int(z[a]))*3**a;b-=int(spec['costs'][a])
