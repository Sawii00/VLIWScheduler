import json


class Bundle:
    def __init__(self, alu0, alu1, mul, mem, br):
        assert alu0.opcode in ["add", "sub", "mov"] or alu0.opcode == "nop"
        assert alu1.opcode in ["add", "sub", "mov"] or alu1.opcode == "nop"
        assert mul.opcode == "mulu" or mul.opcode == "nop"
        assert mem.opcode in ["ld", "st"] or mem.opcode == "nop"
        assert br.opcode in ["loop", "loop.pip"] or br.opcode == "nop"
        self.bundle = (alu0, alu1, mul, mem, br)


class Instruction:
    def __init__(self, pc, opcode, dest, op1, op2):
        self.pc = pc
        self.opcode = opcode
        self.dest = dest
        self.op1 = op1
        self.op2 = op2

    def to_string(self):
        if self.opcode in ["ld", "st"]:
            return f"{self.opcode} {self.dest}, {self.op2}({self.op1})"
        elif self.opcode in ["loop", "loop.pip"]:
            return f"{self.opcode} {self.dest}"
        elif self.opcode == "nop":
            return "nop"
        elif self.opcode == "mov":
            return f"{self.opcode} {self.dest}, {self.op1}"
        else:
            return f"{self.opcode} {self.dest}, {self.op1}, {self.op2}"


class DepTableEntry:
    def __init__(self, id, opcode, destination):
        self.id = id
        self.opcode = opcode
        self.destination = destination
        self.local_dep = []
        self.interloop_dep = []
        self.invariant_dep = []
        self.post_dep = []

    def add_local_dep(self, reg, id):
        self.local_dep.append((reg, id))

    def add_interloop_dep(self, reg, id):
        self.interloop_dep.append((reg, id))

    def add_invariant_dep(self, reg, id):
        self.invariant_dep.append((reg, id))

    def add_postloop_dep(self, reg, id):
        self.post_dep.append((reg, id))

    def to_string(self):
        empty = "__________"
        return f"{self.id} ({self.opcode}) - {self.destination if not self.opcode in ['st', 'loop', 'loop.pip'] else '___'} - {self.local_dep if len(self.local_dep) > 0 else empty}  - {self.interloop_dep if len(self.interloop_dep) > 0 else empty} - {self.invariant_dep if len(self.invariant_dep) > 0 else empty} - {self.post_dep if len(self.post_dep) > 0 else empty} "


class Scheduler:
    def __init__(self, filename, dump_to_file=False):
        self.filename = filename
        self.final_schedule = []

        self.code = self.__parse_json()
        self.dep_table = self.__compute_deps()
        for dep in self.dep_table:
            print(dep.to_string())
        self.__schedule()
        self.__register_rename()
        if dump_to_file:
            self.dump_json(f"{filename[:filename.find('.json')]}_out.json")
        else:
            print(self.get_schedule())

    def __get_latency(self, opcode):
        return 3 if opcode == "mulu" else 1

    '''
        st has the immediate as destination register
    '''
    def __parse_json(self):
        output = []
        with open(self.filename, "r") as file:
            for PC, instruction in enumerate(json.load(file)):
                opcode = instruction.split(" ")[0].strip()
                registers = instruction[instruction.find(" "):].split(",")
                if opcode == "addi": opcode = "add"
                elif opcode == "nop":
                    output.append(Instruction(PC, opcode, None, None, None))
                    continue
                elif opcode == "ld":
                    destination_register = (registers[0].strip())
                    operand_1 = registers[1].strip()[registers[1].find("("): registers[1].find(")") - 1]
                    operand_2 = (registers[1].strip().split("(")[0])
                    output.append(Instruction(PC, opcode, destination_register, operand_1, operand_2))
                    continue
                elif opcode == "st":
                    operand_1 = (registers[0].strip())
                    operand_2  = registers[1].strip()[registers[1].find("("): registers[1].find(")") - 1]
                    destination_register= (registers[1].strip().split("(")[0])
                    output.append(Instruction(PC, opcode, destination_register, operand_1, operand_2))
                    continue
                elif opcode in ["loop", "loop.pip"]:
                    output.append(Instruction(PC, opcode, registers[0].strip(), None, None))
                    continue
                elif opcode == "mov":
                    destination_register = (registers[0].strip())
                    operand_1 = registers[1].strip()
                    output.append(Instruction(PC, opcode, destination_register, operand_1, None))
                    continue

                destination_register = (registers[0].strip())
                operand_1 = (registers[1].strip())
                operand_2 = (registers[2].strip())
                output.append(Instruction(PC, opcode, destination_register, operand_1, operand_2))
        return output

    def __compute_deps(self):
        dep_table = []
        loop_start = -1
        loop_end = -1
        for instr in self.code:
            if instr.opcode in ["loop", "loop.pip"]:
                loop_end = instr.pc
                loop_start = int(instr.dest)
        assert loop_start != -1 and loop_end != -1

        # local + post + loop invariant
        # This will introduce wrong dependencies classified as invariants which will have to be fixed when considering
        # interloop dependencies
        for instr in self.code:
            deps = []
            found_1 = False
            found_2 = False
            for i in range(len(dep_table) - 1, -1, -1):
                entry = dep_table[i]
                #if entry.id >= instr.pc:break
                if entry.opcode == "st":continue
                if not found_1 and entry.destination == instr.op1:
                    deps.append((instr.op1, entry.id))
                    found_1 = True
                if not found_2 and entry.destination == instr.op2:
                    deps.append((instr.op2, entry.id))
                    found_2 = True
                if found_1 and found_2:break

            dep = DepTableEntry(instr.pc, instr.opcode, instr.dest)
            for op_dep in deps:
                if instr.pc < loop_start: dep.add_local_dep(*op_dep)
                elif instr.pc < loop_end:
                    if op_dep[1] < loop_start: dep.add_invariant_dep(*op_dep)
                    else: dep.add_local_dep(*op_dep)
                else:
                    if op_dep[1] < loop_start: dep.add_invariant_dep(*op_dep)
                    elif op_dep[1] < loop_end: dep.add_postloop_dep(*op_dep)
                    else: dep.add_local_dep(*op_dep)
            dep_table.append(dep)

        # Inter-loop Dependencies
        for i in range(loop_start, loop_end):
            instr = self.code[i]
            found_1 = False
            found_2 = False
            deps = []
            for j in range(loop_end, i - 1, -1):
                entry = dep_table[j]
                if entry.opcode == "st": continue
                if not found_1 and entry.destination == instr.op1:
                    deps.append((instr.op1, entry.id))
                    found_1 = True
                if not found_2 and entry.destination == instr.op2:
                    deps.append((instr.op2, entry.id))
                    found_2 = True
                if found_1 and found_2: break

            for op_dep in deps:
                dep_table[i].add_interloop_dep(*op_dep)
                remove_obj = []
                for inv_dep in dep_table[i].invariant_dep:
                    if inv_dep[0] == op_dep[0]:
                        dep_table[i].add_interloop_dep(*inv_dep)
                        remove_obj.append(inv_dep)
                for obj in remove_obj:
                    dep_table[i].invariant_dep.remove(obj)

        return dep_table


    def __schedule(self):
        pass

    def __register_rename(self):
        pass

    def get_schedule(self):
        return self.final_schedule

    def dump_json(self, filename):
        with open(filename, "w") as file:
            json.dump(self.final_schedule, file, indent=2)
