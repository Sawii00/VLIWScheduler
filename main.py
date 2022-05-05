from scheduler import Scheduler
from VLIWCompiler import VLIWCompiler
import json

#sched = Scheduler("tests_3.json")
file = open("errors/code_error_0_1.json")
sched_pip = Scheduler("errors/code_error_0_1.json", pip=True, dump_to_file=False)
vliw_compiler_looppip = VLIWCompiler("test.json")
res2 = vliw_compiler_looppip.compile(json.dumps(json.load(file)))

# Cannot use LC or EC or any predicate as operand
