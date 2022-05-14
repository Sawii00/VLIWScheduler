import random
import json
import scheduler as sawo
import main_gianlu
from json_encoder import CustomEncoder
import vliw470


class Fuzzer:
    def __init__(self):
        self.instructions = ["add", "addi", "sub", "mulu", "ld", "st", "mov"]
        self.registers = [f"x{i}" for i in range(1, 32)]

    def pick_random_instr(self):
        res = 0
        instruction = random.choice(self.instructions)
        if instruction not in ["ld", "st", "mov"]:
            dest = random.choice(self.registers)
            first_op = random.choice(self.registers)
            second_op = 0
            if instruction == "addi":
                second_op = random.randint(1, 30)
            else:
                second_op = random.choice(self.registers)
            res = (f"{instruction} {dest}, {first_op}, {second_op}")
        elif instruction == "ld":
            dest = random.choice(self.registers)
            offset = random.randint(0, 30)
            source = random.choice(self.registers)
            res = (f"{instruction} {dest}, {offset}({source})")
        elif instruction == "st":
            addr = random.choice(self.registers)
            offset = random.randint(0, 30)
            source = random.choice(self.registers)
            res = (f"{instruction} {source}, {offset}({addr})")
        elif instruction == "mov":
            dest = random.choice(self.registers)
            if random.random() > 0.5:
                source = random.choice(self.registers)
            else:
                source = random.randint(5, 50)
            res = (f"{instruction} {dest}, {source}")
        return res

    def fix_test(self, test):
        destinations = set()
        for instr in test:
            opcode, dest = instr.split(",")[0].split(" ")
            if opcode not in ["st", "loop"] and dest not in ["EC", "LC"]:
                destinations.add(dest)

        for i, instr in enumerate(destinations):
            test.append(f"st {instr}, 0({instr})")
        return test

    def generate_tests(self, n, pre_loop_max_length, loop_max_length, post_loop_max_length):
        tests = []
        for _ in range(n):
            code = []
            for _ in range(random.randint(3, pre_loop_max_length)):
                code.append(self.pick_random_instr())
            code.append(f"mov LC, {random.randint(5, 20)}")
            pre_loop = len(code)
            for _ in range(random.randint(3, loop_max_length)):
                code.append(self.pick_random_instr())
            code.append(f"loop {pre_loop}") 
            for _ in range(random.randint(3, post_loop_max_length)):
                code.append(self.pick_random_instr())
            code = self.fix_test(code)
            tests.append(code)
        return tests

    def test(self, n=10, pre_length=3, loop_length=5, post_length=3):
        tests = self.generate_tests(n, pre_length, loop_length, post_length)
        errors = []
        for test in tests:
            gianlu = main_gianlu.do_stuff(json.dumps(test))
            for i, pip in enumerate([False, True]):
                sim1 = sawo.Scheduler(test, pip)
                res2 = vliw470.simulate(json.loads(gianlu[i]))
                res1 = vliw470.simulate(json.loads(sim1.get_schedule_dump()))
                if json.dumps(res1.data, sort_keys=True) != json.dumps(res2.data, sort_keys=True):
                    print("Mismatch found")
                    errors.append((test, json.loads(sim1.get_schedule_dump()), json.loads(gianlu[i])))
        return errors


fuzzer = Fuzzer()
errors = fuzzer.test(1000)

for i, error in enumerate(errors):
    with open(f"errors/code_error_{i}_1.json", "w") as file:
        json.dump(error[0], file, indent=2)
    with open(f"errors/error_{i}_1.json", "w") as file:
        json.dump(error[1], file, indent=2, cls=CustomEncoder)
    with open(f"errors/error_{i}_2.json", "w") as file:
        json.dump(error[2], file, indent=2, cls=CustomEncoder)
