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


class Scheduler:
    def __init__(self, filename, dump_to_file=False):
        self.filename = filename
        self.final_schedule = []

        self.code = self.__parse_json()
        self.dep_table = self.__compute_deps()
        self.__schedule()
        self.__register_rename()
        if dump_to_file:
            self.dump_json(f"{filename[:filename.find('.json')]}_out.json")
        else:
            print(self.get_schedule())

    def __get_latency(self, opcode):
        return 3 if opcode == "mulu" else 1

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
                elif opcode in ["ld", "st"]:
                    destination_register = (registers[0].strip())
                    operand_1 = registers[1].strip()[registers[1].find("("): registers[1].find(")") - 1]
                    operand_2 = (registers[1].strip().split("(")[0])
                    output.append(Instruction(PC, opcode, destination_register, operand_1, operand_2))
                    continue
                elif opcode in ["loop", "loop.pip"]:
                    output.append(Instruction(PC, opcode, registers[0], None, None))
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
        return None

    def __schedule(self):
        pass

    def __register_rename(self):
        pass

    def get_schedule(self):
        return self.final_schedule

    def dump_json(self, filename):
        with open(filename, "w") as file:
            json.dump(self.final_schedule, file, indent=2)
