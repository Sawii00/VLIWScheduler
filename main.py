from scheduler import Scheduler
from VLIWCompiler import VLIWCompiler
import json

#sched = Scheduler("tests_3.json")
file = open("errors/test2.json")
sched_pip = Scheduler("handout.json", pip=False, dump_to_file=True)
vliw_compiler_looppip = VLIWCompiler("test.json")
res2 = vliw_compiler_looppip.compile(json.dumps(json.load(file)))

# Cannot use LC or EC or any predicate as operand
