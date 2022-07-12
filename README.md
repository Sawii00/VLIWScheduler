# A Cycle-by-Cycle Scheduler for a VLIW Processor
Final submission for Lab2 of CS-470 (Advanced Computer Architecture) at EPFL. 

The goal was to write a Scheduler for a VLIW processor based on a RISCV ISA, but providing Itanium features for extracting parallelism such as Rotating Registers. 

The scheduler has to perform both scheduling and register renaming to produce an optimized output, either by using loop-only instructions or by attempting to pipeline the circuit through the loop.pip instruction and Rotating Registers.

In order to run the scheduler on a test, provide the name as command-line argument.

python main.py filename.json

The two required schedules as json outputs will be placed in the same folder of the test and will have the following names:
- **loop**: filename_out_loop.json
- **loop-pip**: filename_out_loop_pip.json
