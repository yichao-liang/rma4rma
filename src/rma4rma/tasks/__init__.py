from rma4rma.tasks.peg_insertion import PegInsertionRMA
from rma4rma.tasks.pick_cube import PickCubeRMA
from rma4rma.tasks.pick_single import PickSingleYCBRMA
from rma4rma.tasks.stack_cube import StackCubeRMA
from rma4rma.tasks.turn_faucet import TurnFaucetRMA

gym_task_map = {
    "PickCube": PickCubeRMA,
    "PickSingleYCB": PickSingleYCBRMA,
    "StackCube": StackCubeRMA,
    "PegInsert": PegInsertionRMA,
    "TurnFaucet": TurnFaucetRMA,
}
