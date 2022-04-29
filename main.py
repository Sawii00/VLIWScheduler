from scheduler import Scheduler

sched = Scheduler("handout.json")

# TODO: see what happens if you have invariant + local in post-loop --> we can fix by first fixing interloop schedule and only after schedule the post-loop
# Issue with operand register that were never written before when renaming the destinations (look screenshot1)

#TODO: fix register available routine
