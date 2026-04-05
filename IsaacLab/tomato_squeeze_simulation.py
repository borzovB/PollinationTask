# FRANKA выжимает помидоры и измеряет усилие
# запуск:
# isaaclab.bat -p tomato_squeeze_simulation.py

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "width": 1600, "height": 900})

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import RMPFlowController
from omni.isaac.core.utils.types import ArticulationAction

# Создание мира

world = World(
    stage_units_in_meters=1.0,
    physics_dt=1/240.0,
    rendering_dt=1/60.0
)

world.scene.add_default_ground_plane()

# Создание робота

franka = world.scene.add(
    Franka(
        prim_path="/World/Franka",
        name="franka",
        position=np.array([0.0, 0.0, 0.0])
    )
)

# Создание помидора

TOMATO_POS = np.array([0.45, 0.0, 0.033])

tomato = world.scene.add(
    DynamicSphere(
        prim_path="/World/Tomato",
        name="tomato",
        position=TOMATO_POS,
        radius=0.033,
        mass=0.113,
        color=np.array([0.9, 0.1, 0.1]),
    )
)

# Сброс

world.reset()

franka.gripper.open()

for _ in range(50):
    world.step(render=False)

# Контроллер

controller = RMPFlowController(
    name="rmpflow",
    robot_articulation=franka
)

# Выставляем позиции суставов манипуляторов

TARGET_APPROACH = np.array([0.45, 0.0, 0.12])
TARGET_GRASP = np.array([0.45, 0.0, 0.045])
TARGET_LIFT = np.array([0.45, 0.0, 0.35])

orient_down = np.array([0.0, 1.0, 0.0, 0.0])

# Фазы манипулятора

PH_APPROACH = 400
PH_LOWER = 800
PH_WAIT = 1000
PH_GRIP = 1400
PH_HOLD = 1700
PH_LIFT = 2200

TOTAL = PH_LIFT

# Силовые параметры

lever_arm = 0.06

SAFE_FORCE = 5
RISK_FORCE = 10

# Логирование

print(f"\n{'step':>5} {'phase':<10} {'force(N)':>10} {'status'}")
print("-"*45)

# Запуск симуляции

for step in range(TOTAL):

    if step < PH_APPROACH:

        target = TARGET_APPROACH
        grip = 0.04
        phase = "approach"

    elif step < PH_LOWER:

        target = TARGET_GRASP
        grip = 0.04
        phase = "lower"

    elif step < PH_WAIT:

        target = TARGET_GRASP
        grip = 0.04
        phase = "wait"

    elif step < PH_GRIP:

        target = TARGET_GRASP

        prog = (step - PH_WAIT) / (PH_GRIP - PH_WAIT)
        grip = max(0.04 - 0.04 * prog, 0.0)

        phase = "squeeze"

    elif step < PH_HOLD:

        target = TARGET_GRASP
        grip = 0.0
        phase = "hold"

    else:

        target = TARGET_LIFT
        grip = 0.0
        phase = "lift"

    # Движение руки

    actions = controller.forward(
        target_end_effector_position=target,
        target_end_effector_orientation=orient_down
    )

    franka.apply_action(actions)

    # Движение гриппера

    franka.apply_action(
        ArticulationAction(
            joint_positions=np.array([grip, grip]),
            joint_indices=np.array([7, 8])
        )
    )

    world.step(render=True)

    # Измерение усилия

    efforts = franka.get_measured_joint_efforts()

    left_torque = abs(efforts[7])
    right_torque = abs(efforts[8])

    grip_force = (left_torque + right_torque) / (2 * lever_arm)

    # Статус

    if grip_force < SAFE_FORCE:
        status = "safe"
    elif grip_force < RISK_FORCE:
        status = "risky"
    else:
        status = "crushed"

    if step % 100 == 0:

        print(
            f"{step:5d} {phase:<10} {grip_force:10.2f} {status}"
        )

# Закрытие

simulation_app.close()