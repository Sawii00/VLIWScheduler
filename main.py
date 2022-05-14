from scheduler import Scheduler
import sys

if len(sys.argv) != 2:
    print("Usage: python main.py filename")
    exit(-1)

sched_loop = Scheduler(sys.argv[1], pip=False, dump_to_file=True)
sched_loop_pip = Scheduler(sys.argv[1], pip=True , dump_to_file=True)

