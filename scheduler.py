import json


class Bundle:
    def __init__(self, pc):
        self.pc = pc
        self.alu0 = None
        self.alu1 = None
        self.mul = None
        self.mem = None
        self.br = None

    def find(self, pc):
        if self.alu0 is not None and self.alu0.pc == pc:return self.alu0
        if self.alu1 is not None and self.alu1.pc == pc:return self.alu1
        if self.mul is not None and self.mul.pc == pc:return self.mul
        if self.mem is not None and self.mem.pc == pc:return self.mem
        if self.br is not None and self.br.pc == pc:return self.br
        return None

    def set_alu0(self, alu0):
        assert alu0.opcode in ["add", "sub", "mov"] or alu0.opcode == "nop"
        self.alu0 = alu0

    def set_alu1(self, alu1):
        assert alu1.opcode in ["add", "sub", "mov"] or alu1.opcode == "nop"
        self.alu1 = alu1

    def set_mul(self, mul):
        assert mul.opcode == "mulu" or mul.opcode == "nop"
        self.mul = mul

    def set_mem(self, mem):
        assert mem.opcode in ["ld", "st"] or mem.opcode == "nop"
        self.mem = mem

    def set_br(self, br):
        assert br.opcode in ["loop", "loop.pip"] or br.opcode == "nop"
        self.br = br

    def schedule_instr(self, instr):
        if instr.opcode in ["add", "sub", "mov"]:
            if self.alu0 is None:
                self.alu0 = instr
                return True
            if self.alu1 is None:
                self.alu1 = instr
                return True
            return False
        elif instr.opcode == "mulu" and self.mul is None:
            self.mul = instr
            return True
        elif instr.opcode in ["st", "ld"] and self.mem is None:
            self.mem = instr
            return True
        elif instr.opcode.startswith("loop"):
            if self.br is None:
                self.br = instr
                return True
        return False

    def to_string(self):
        alu0 = f"{chr(ord('A') + self.alu0.pc)} ({self.alu0.opcode} {self.alu0.dest}, {self.alu0.op1}, {self.alu0.op2})" if self.alu0 is not None else None
        alu1 = f"{chr(ord('A') + self.alu1.pc)} ({self.alu1.opcode} {self.alu1.dest}, {self.alu1.op1}, {self.alu1.op2})" if self.alu1 is not None else None
        mul = f"{chr(ord('A') + self.mul.pc)} ({self.mul.opcode} {self.mul.dest}, {self.mul.op1}, {self.mul.op2})" if self.mul is not None else None
        mem = f"{chr(ord('A') + self.mem.pc)} ({self.mem.opcode} {self.mem.dest}, {self.mem.op1}, {self.mem.op2})" if self.mem is not None else None
        br = f"{chr(ord('A') + self.br.pc)} ({self.br.opcode})" if self.br is not None else None
        return f"Bundle({self.pc}): {alu0} | {alu1} | {mul} | {mem} | {br}\n"

    def __repr__(self):
        return self.to_string()

    def __str__(self):
        return self.to_string()


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

    def __repr__(self):
        return self.to_string()

    def __str__(self):
        return self.to_string()


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

    def get_all_deps(self):
        return self.local_dep + self.interloop_dep + self.invariant_dep + self.post_dep

    def to_string(self):
        empty = "__________"
        return f"{self.id} ({self.opcode}) - {self.destination if not self.opcode in ['st', 'loop', 'loop.pip'] else '___'} - {self.local_dep if len(self.local_dep) > 0 else empty}  - {self.interloop_dep if len(self.interloop_dep) > 0 else empty} - {self.invariant_dep if len(self.invariant_dep) > 0 else empty} - {self.post_dep if len(self.post_dep) > 0 else empty} "

    def __repr__(self):
        return self.to_string()

    def __str__(self):
        return self.to_string()


class Scheduler:
    def __init__(self, filename, dump_to_file=False):
        self.filename = filename
        self.final_schedule = []
        self.loop_start = 0
        self.loop_end = 0
        self.scheduled_slot = []

        self.code = self.__parse_json()
        self.dep_table = self.__compute_deps()
        for dep in self.dep_table:
            print(dep.to_string())
        # WE ARE NOT CHECKING THAT A LOOP EVEN EXISTS IN CODE (WE ASSUME IT IS THERE)
        if self.__is_loop_program():
            self.__schedule_loop()
            self.__register_rename_loop()
        else:
            self.__schedule_loop_pip()
            self.__register_rename_loop_pip()

        if dump_to_file:
            self.dump_json(f"{filename[:filename.find('.json')]}_out.json")
        else:
            print(self.get_schedule())
            print(self.ii)

    def __get_latency(self, opcode):
        return 3 if opcode == "mulu" else 1

    def __is_loop_program(self):
        for instr in self.code:
            if instr.opcode == "loop":
                return True
        return False

    '''
        st has the immediate as destination register
    '''

    def __parse_json(self):
        output = []
        with open(self.filename, "r") as file:
            for PC, instruction in enumerate(json.load(file)):
                opcode = instruction.split(" ")[0].strip()
                registers = instruction[instruction.find(" "):].split(",")
                if opcode == "addi":
                    opcode = "add"
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
                    operand_2 = registers[1].strip()[registers[1].find("("): registers[1].find(")") - 1]
                    destination_register = (registers[1].strip().split("(")[0])
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
        self.loop_start = loop_start
        self.loop_end = loop_end

        # local + post + loop invariant
        # This will introduce wrong dependencies classified as invariants which will have to be fixed when considering
        # interloop dependencies
        for instr in self.code:
            deps = []
            found_1 = False
            found_2 = False
            for i in range(len(dep_table) - 1, -1, -1):
                entry = dep_table[i]
                # if entry.id >= instr.pc:break
                if entry.opcode == "st": continue
                if not found_1 and entry.destination == instr.op1:
                    deps.append((instr.op1, entry.id))
                    found_1 = True
                if not found_2 and entry.destination == instr.op2:
                    deps.append((instr.op2, entry.id))
                    found_2 = True
                if found_1 and found_2: break

            dep = DepTableEntry(instr.pc, instr.opcode, instr.dest)
            for op_dep in deps:
                if instr.pc < loop_start:
                    dep.add_local_dep(*op_dep)
                elif instr.pc < loop_end:
                    if op_dep[1] < loop_start:
                        dep.add_invariant_dep(*op_dep)
                    else:
                        dep.add_local_dep(*op_dep)
                else:
                    if op_dep[1] < loop_start:
                        dep.add_invariant_dep(*op_dep)
                    elif op_dep[1] < loop_end:
                        dep.add_postloop_dep(*op_dep)
                    else:
                        dep.add_local_dep(*op_dep)
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

    def __fix_schedule(self, scheduled_slots):
        # compute delta
        delta = -1
        to_be_fixed = False
        end_loop = scheduled_slots[self.loop_end]
        start_loop = scheduled_slots[self.loop_start]
        for entry in self.dep_table:
            interloops = entry.interloop_dep
            for dep in interloops:
                s_c = scheduled_slots[entry.id]
                lambda_p = self.__get_latency(entry.opcode)
                s_p = scheduled_slots[dep[1]]
                if s_p + lambda_p > s_c + self.ii:
                    to_be_fixed = True
                    curr_delta = end_loop - s_p + s_c - start_loop + 1
                    if curr_delta > delta: delta = curr_delta
        for i in range(delta):
            self.final_schedule.insert(end_loop + 1, Bundle(end_loop + 1))
            for j in range(len(scheduled_slots)):
                if scheduled_slots[j] > scheduled_slots[self.loop_end]: scheduled_slots[j] += 1
            scheduled_slots[self.loop_end] += 1
            self.ii += 1
        # Swapping the loop position
        if to_be_fixed:
            self.final_schedule[scheduled_slots[self.loop_end]].set_br(self.final_schedule[scheduled_slots[self.loop_end] - delta].br)
            self.final_schedule[scheduled_slots[self.loop_end] - delta].br = None
            for i in range(len(self.final_schedule)):
                self.final_schedule[i].pc = i

    def __schedule_loop(self):
        pc = 0
        scheduled_slot = [-10 for i in range(len(self.dep_table))]
        loop_reached = False
        end_loop_reached = False
        while scheduled_slot.count(-10) != 0:
            self.final_schedule.append(Bundle(pc))
            for i, instr in enumerate(self.dep_table):
                if i == self.loop_start and not loop_reached:
                    if scheduled_slot[:self.loop_start].count(-10) == 0:
                        loop_reached = True
                    break
                if i == self.loop_end and not end_loop_reached:
                    if scheduled_slot[self.loop_start:self.loop_end].count(-10) == 0:
                        end_loop_reached = True
                    if not end_loop_reached: break
                if scheduled_slot[i] != -10: continue
                deps = instr.get_all_deps()
                earliest_time = -10
                for dep in deps:
                    start_time = scheduled_slot[dep[1]] + self.__get_latency(self.dep_table[dep[1]].opcode)
                    if start_time > earliest_time:
                        earliest_time = start_time
                if earliest_time <= pc:
                    res = self.final_schedule[pc].schedule_instr(self.code[instr.id])
                    if res:
                        scheduled_slot[i] = pc
                if i == self.loop_end: break  # to split bb2 from bb1
            pc += 1
        self.ii = scheduled_slot[self.loop_end] - scheduled_slot[self.loop_start] + 1

        self.__fix_schedule(scheduled_slot)
        self.scheduled_slot = scheduled_slot
        # TODO: update loop target

    def __register_rename_loop(self):
        # Destination Register Allocation
        curr_reg = 1
        for bundle in self.final_schedule:
            for instr in [bundle.alu0, bundle.alu1, bundle.mul, bundle.mem]:
                if instr is not None and instr.dest not in ["LC", "EC"] and instr.opcode != "st":
                    instr.dest = f"x{curr_reg}"
                    curr_reg += 1
        # Operand Linking
        interloop_missing = set()
        for bundle in self.final_schedule:
            for instr in [bundle.alu0, bundle.alu1, bundle.mul, bundle.mem]:
                if instr is not None:
                    for dep in self.dep_table[instr.pc].local_dep + self.dep_table[instr.pc].invariant_dep + self.dep_table[instr.pc].post_dep:
                        if instr.op1 == dep[0]:
                            instr.op1 = self.final_schedule[self.scheduled_slot[dep[1]]].find(dep[1]).dest
                        if instr.op2 == dep[0]:
                            instr.op2 = self.final_schedule[self.scheduled_slot[dep[1]]].find(dep[1]).dest
                    dep = 0
                    if len(self.dep_table[instr.pc].interloop_dep) == 2:
                        deps_sorted = sorted(self.dep_table[instr.pc].interloop_dep, key=lambda x: x[1])
                        dep = deps_sorted[0]
                        interloop_missing.add((deps_sorted[1][1], self.final_schedule[self.scheduled_slot[deps_sorted[1][1]]].find(deps_sorted[1][1]).dest, self.final_schedule[self.scheduled_slot[dep[1]]].find(dep[1]).dest))
                    elif len(self.dep_table[instr.pc].interloop_dep) == 1:
                        dep = self.dep_table[instr.pc].interloop_dep[0]
                    else:
                        continue
                    if instr.op1 == dep[0]:
                        instr.op1 = self.final_schedule[self.scheduled_slot[dep[1]]].find(dep[1]).dest
                    if instr.op2 == dep[0]:
                        instr.op2 = self.final_schedule[self.scheduled_slot[dep[1]]].find(dep[1]).dest
        # Interloop Handling
        print(interloop_missing)
        increments = 0
        for dep in interloop_missing:
            curr_pos = self.scheduled_slot[self.loop_end]
            start_time = self.scheduled_slot[dep[0]] + self.__get_latency(self.dep_table[dep[0]].opcode)
            while True:
                if start_time <= curr_pos:
                    res = self.final_schedule[curr_pos].schedule_instr(Instruction(ord("Z") - ord("A"), "mov", dep[2], dep[1], None))
                    if res:
                        break
                else:
                    self.final_schedule.insert(self.scheduled_slot[self.loop_end] + 1, Bundle(self.scheduled_slot[self.loop_end] + 1))
                    for j in range(len(self.scheduled_slot)):
                        if self.scheduled_slot[j] > self.scheduled_slot[self.loop_end]: self.scheduled_slot[j] += 1
                    increments += 1
                    self.ii += 1
                    self.final_schedule[self.scheduled_slot[self.loop_end] + increments].set_br(self.final_schedule[self.scheduled_slot[self.loop_end] + increments - 1].br)
                    self.final_schedule[self.scheduled_slot[self.loop_end] + increments - 1].br = None
                    for i in range(len(self.final_schedule)):
                        self.final_schedule[i].pc = i
                curr_pos += 1
        # to change the perceived loop position at the end because otherwise mov instructions will try to schedule from
        # the current loop position that can be already moved down if previous moves caused it to move
        self.scheduled_slot[self.loop_end] += increments

    def __schedule_loop_pip(self):
        pass

    def __register_rename_loop_pip(self):
        pass

    def get_schedule(self):
        return self.final_schedule

    def dump_json(self, filename):
        with open(filename, "w") as file:
            json.dump(self.final_schedule, file, indent=2)
