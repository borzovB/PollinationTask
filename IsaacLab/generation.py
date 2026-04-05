# Генератор датасета захватов — Isaac Sim headless
# Запуск: _isaac_sim\isaaclab.bat -p generation.py

# Версия номер 2
import os
import random
import numpy as np
import time
import csv

# Симуляция без показа демонтрации робота
from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": True,
    "disable_viewport_updates": True
})

# Задается физика, введен цент масс
from omni.isaac.core.physics_context import PhysicsContext
import omni
import omni.physx
from omni.isaac.core import World
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import RMPFlowController
from omni.isaac.core.utils.types import ArticulationAction
from pxr import (UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Gf, Vt, Sdf)

# КОНСТАНТЫ И ШАБЛОНЫ
# Цвет объектов
COLOR_RGB = {
    "gray": [0.50, 0.50, 0.50],
    "white": [0.98, 0.96, 0.92],
    "red": [0.90, 0.10, 0.10],
    "green": [0.10, 0.60, 0.10],
    "lightblue": [0.80, 0.92, 1.00],
    "silver": [0.80, 0.82, 0.85],
}

# Набор данных для проведения калибровки и эмпирической
# корректировки, сформированный на основе справочных источников и экспериментальных исследований

TEMPLATES = [
    dict(
        name="Stone", shape="cube", density=2600,
        sx=0.01, sy=0.01, sz=0.01, E=5e10, nu=0.25, mu=0.80,
        stress=8.5e7, safe_force=100, color="gray",
        transparency="opaque", hardness=7.0
    ),
    dict(
        name="Chicken egg", shape="ellipsoid", density=2200,
        sx=0.07, sy=0.04, sz=0.04, E=5e9, nu=0.307, mu=0.60,
        stress=5e4, safe_force=10, color="white",
        transparency="opaque", hardness=3.0
    ),
    dict(
        name="Tomato", shape="sphere", density=1000,
        sx=0.07, sy=0.07, sz=0.07, E=5e4, nu=0.30, mu=0.50,
        stress=3e4, safe_force=6, color="red",
        transparency="opaque", hardness=0.5
    ),
    dict(
        name="Cucumber", shape="cylinder", density=1000,
        sx=0.17, sy=0.04, sz=0.04, E=2e5, nu=0.35, mu=0.67,
        stress=3e4, safe_force=6, color="green",
        transparency="opaque", hardness=0.3
    ),
    dict(
        name="Rubber ball", shape="sphere", density=1100,
        sx=0.07, sy=0.07, sz=0.07, E=1e7, nu=0.49, mu=1.00,
        stress=4e4, safe_force=8, color="red",
        transparency="opaque", hardness=0.1
    ),
    dict(
        name="Plastic bottle", shape="cylinder", density=1340,
        sx=0.28, sy=0.09, sz=0.09, E=2.5e9, nu=0.39, mu=0.80,
        stress=3e5, safe_force=60, color="lightblue",
        transparency="transparent", hardness=2.0
    ),
    dict(
        name="Tin can", shape="cylinder", density=7700,
        sx=0.11, sy=0.08, sz=0.08, E=2e11, nu=0.35, mu=0.86,
        stress=4.5e5, safe_force=90, color="silver",
        transparency="opaque", hardness=4.5
    ),
]

# Число экспериментов
NUM_EXPERIMENTS = 5

# Получение пути к родительской папке
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(PROJECT_DIR, "source", "standalone", "grasp_dataset.csv")
# Плечо силы для измерения силы давления манипулятора на объект
LEVER_ARM = 0.015

# Гравитационная постоянная
G = 9.81
# Путь к
OBJ_PATH = "/World/GraspObject"

# ГЕОМЕТРИЧЕСКИЕ ФУНКЦИИ

def compute_volume(shape, sx, sy, sz):
    # Вычисляет объём объекта по форме и размерам
    if shape == "cube":
        return sx * sy * sz
    elif shape == "sphere":
        return (4/3) * np.pi * (sy/2)**3
    elif shape == "cylinder":
        return np.pi * (sy/2)**2 * sx
    return sx * sy * sz


def compute_surface(shape, sx, sy, sz):
    # Вычисляет площадь поверхности объекта
    if shape == "cube":
        return 2 * (sx*sy + sy*sz + sx*sz)
    elif shape == "sphere":
        return 4 * np.pi * (sy/2)**2
    elif shape == "cylinder":
        return 2 * np.pi * (sy/2) * (sy/2 + sx)
    return 6 * sx * sy


def get_last_id(nameFile):
    if not os.path.isfile(nameFile):
        return 0

    try:
        with open(nameFile, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            lastId = 0
            for row in reader:
                currentId = int(row.get('exp_id',0))
                lastId = max(lastId, currentId)

            return lastId
    except Exception as e:
        print("Ошибка чтения файла")
        return  0

# СОЗДАНИЕ ОБЪЕКТОВ ЧЕРЕЗ USD

def create_usd_sphere(stage, path: str, pos: np.ndarray, radius: float,
                      mass: float, color: list, mu: float):
    # Создаёт физическую сферу напрямую в USD
    # Удаляем если уже есть
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)

    # Создаём Xform-контейнер
    xform = UsdGeom.Xform.Define(stage, path)
    UsdGeom.XformCommonAPI(xform).SetTranslate(
        Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
    )

    # Геометрия сферы
    sphere_path = path + "/sphere"
    sphere = UsdGeom.Sphere.Define(stage, sphere_path)
    sphere.GetRadiusAttr().Set(float(radius))
    sphere.GetDisplayColorAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(*[float(c) for c in color])])
    )

    # Физика
    prim = stage.GetPrimAtPath(path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(sphere_path))
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(mass))

    # случайное смещение центра масс
    com_offset = np.random.uniform(-0.005, 0.005, 3)
    mass_api.CreateCenterOfMassAttr(
        Gf.Vec3f(float(com_offset[0]), float(com_offset[1]), float(com_offset[2]))
    )

    PhysxSchema.PhysxCollisionAPI.Apply(stage.GetPrimAtPath(sphere_path))

    # Материал трения
    mat_path = path + "/PhysMat"
    mat_prim = stage.DefinePrim(mat_path, "Material")
    pm = UsdPhysics.MaterialAPI.Apply(mat_prim)
    pm.CreateStaticFrictionAttr(float(mu))
    pm.CreateDynamicFrictionAttr(float(mu * 0.9))
    pm.CreateRestitutionAttr(0.1)

    UsdShade.MaterialBindingAPI.Apply(
        stage.GetPrimAtPath(sphere_path)
    ).Bind(
        UsdShade.Material(mat_prim),
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )
    return prim


def create_usd_cube(stage, path, pos, sx, sy, sz, mass, color):
    # Создаёт физический куб напрямую в USD
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)

    xform = UsdGeom.Xform.Define(stage, path)
    UsdGeom.XformCommonAPI(xform).SetTranslate(
        Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
    )

    cube_path = path + "/cube"
    cube = UsdGeom.Cube.Define(stage, cube_path)
    cube.GetSizeAttr().Set(1.0)
    UsdGeom.XformCommonAPI(cube).SetScale(
        (float(sx), float(sy), float(sz))
    )
    cube.GetDisplayColorAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(*color)])
    )

    prim = stage.GetPrimAtPath(path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(cube_path))
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(mass))

    # случайное смещение центра масс
    com_offset = np.random.uniform(-0.005, 0.005, 3)
    mass_api.CreateCenterOfMassAttr(
        Gf.Vec3f(float(com_offset[0]), float(com_offset[1]), float(com_offset[2]))
    )

    PhysxSchema.PhysxCollisionAPI.Apply(stage.GetPrimAtPath(cube_path))
    return prim


def create_usd_cylinder(stage, path, pos, sx, sy, mass, color):
    # Создаёт физический цилиндр напрямую в USD
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)

    xform = UsdGeom.Xform.Define(stage, path)
    UsdGeom.XformCommonAPI(xform).SetTranslate(
        Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
    )

    cyl_path = path + "/cylinder"
    cyl = UsdGeom.Cylinder.Define(stage, cyl_path)
    cyl.GetRadiusAttr().Set(float(sy/2))
    cyl.GetHeightAttr().Set(float(sx))
    cyl.GetDisplayColorAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(*color)])
    )

    prim = stage.GetPrimAtPath(path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(cyl_path))
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(mass))

    # случайное смещение центра масс
    com_offset = np.random.uniform(-0.005, 0.005, 3)
    mass_api.CreateCenterOfMassAttr(
        Gf.Vec3f(float(com_offset[0]), float(com_offset[1]), float(com_offset[2]))
    )

    PhysxSchema.PhysxCollisionAPI.Apply(stage.GetPrimAtPath(cyl_path))
    return prim


def delete_usd_object(stage, path: str):
    # Удаляет объект из USD
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)

# ИНИЦИАЛИЗАЦИЯ СЦЕНЫ

print("Инициализация сцены...")
# Создаем мир
world = World(stage_units_in_meters=1.0, physics_dt=1/120.0, rendering_dt=None)
world.scene.add_default_ground_plane()

stage = omni.usd.get_context().get_stage()

# Создаем робота
franka = world.scene.add(
    Franka(
        prim_path="/World/Franka",
        name="Franka",
        position=np.array([0.0, 0.0, 0.0])
    )
)
# Перезапуск мира
world.reset()

# Установка позиции пальцев манипулятора
franka.gripper.open()

# Работа симуляции
for _ in range(30):
    world.step(render=False)

# ГЛАВНЫЙ ЦИКЛ ЭКСПЕРИМЕНТОВ

# Переменные в датасете
FIELDS = [
    "exp_id", "object_name", "shape_type", "density", "volume",
    "mass", "size_x", "size_y", "size_z", "surface_area",
    "contact_area", "young_modulus", "poisson_ratio", "friction",
    "stress_limit", "safe_force", "color", "transparency", "hardness",
    "grasp_force_N", "contact_stress", "stress_ratio", "outcome",
    "label_success", "label_crush", "label_drop", "duration_sec",
]

# Рандомизатор
rng = np.random.default_rng(42)
# Засекаем время работы
t_total = time.time()

lastId = get_last_id(OUTPUT_FILE)
print(lastId)
if lastId == 0:
    print("Создан новый файл с заголовками")
else:
    print("Продолжение записи в существующий файл")

print(f"\nЗапуск {NUM_EXPERIMENTS} экспериментов...\n")
print(f"{'exp':>5} {'объект':<18} {'F,Н':>8} {'исход':<10} {'сек':>5}")
print("-" * 55)

start_file = lastId
stop_file = NUM_EXPERIMENTS
# Начало работы симуляции и записи результатов в файл
with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if lastId == 0:
        writer.writeheader()

    # Итерации симуляций
    for exp_id in range(NUM_EXPERIMENTS):
        t0 = time.time()
        exp_idx = exp_id + start_file

        # Выбор и масштабирование шаблона
        tmpl = rng.choice(TEMPLATES)
        scale = float(rng.uniform(0.85, 1.20))

        # Размер объектов
        sx = tmpl["sx"] * scale
        sy = tmpl["sy"] * scale
        sz = tmpl["sz"] * scale

        # Обработка размеров
        if tmpl["shape"] == "sphere":
            sx = sy = sz = tmpl["sy"] * scale
        elif tmpl["shape"] == "cylinder":
            sy = sz = tmpl["sy"] * scale

        # Генерация физических параметров с шумом
        density = float(rng.normal(tmpl["density"], tmpl["density"] * 0.1))
        E = float(rng.normal(tmpl["E"], tmpl["E"] * 0.1))
        nu = float(np.clip(rng.normal(tmpl["nu"], tmpl["nu"] * 0.1), 0.01, 0.499))
        mu = float(np.clip(rng.normal(tmpl["mu"], tmpl["mu"] * 0.1), 0.05, 1.5))
        stress_lim = float(rng.normal(tmpl["stress"], tmpl["stress"] * 0.1))

        # Геометрические параметры
        volume = compute_volume(tmpl["shape"], sx, sy, sz)
        mass = density * volume
        surface = compute_surface(tmpl["shape"], sx, sy, sz)

        # Площадь поверхности касания
        contact_area = 2e-4 * rng.uniform(0.6, 1.4)

        # Создание объекта с радиусом
        radius = sy / 2.0
        obj_pos = np.array([
            0.45 + float(rng.uniform(-0.04, 0.04)),
            float(rng.uniform(-0.06, 0.06)),
            max(radius + 0.005, 0.04),
        ])

        col = COLOR_RGB[tmpl["color"]]

        # Расчёт сил
        F_min_hold = mass * G * 1.3 / (2 * mu + 1e-9)
        F_max_safe = 2 * mu * stress_lim * contact_area - mass * G

        # Создание объекта
        if tmpl["shape"] == "sphere":
            radius = sy / 2
            create_usd_sphere(stage, OBJ_PATH, obj_pos, radius, mass, col, mu)
        elif tmpl["shape"] == "cube":
            create_usd_cube(stage, OBJ_PATH, obj_pos, sx, sy, sz, mass, col)
        elif tmpl["shape"] == "cylinder":
            create_usd_cylinder(stage, OBJ_PATH, obj_pos, sx, sy, mass, col)

        # Сброс физики
        world.reset()
        franka.gripper.open()
        controller = RMPFlowController(
            name="rmpflow",
            robot_articulation=franka
        )

        for _ in range(20):
            world.step(render=False)

        # Эксперимент с захватом
        T_APPROACH = obj_pos + np.array([0, 0, 0.14])
        T_GRASP = obj_pos + np.array([0, 0, 0.02])
        T_LIFT = obj_pos + np.array([0, 0, 0.30])
        orient = np.array([0.0, 1.0, 0.0, 0.0])

        # Счетчик позиций манипулятора
        PH_APPROACH = 200
        PH_LOWER = 300
        PH_WAIT = 400
        PH_GRIP = 500

        # Счетчик силы
        max_force = 0.0
        crushed = False

        # Движение гриппера на 0.04
        for step in range(PH_GRIP):
            if step < PH_APPROACH:
                target, grip = T_APPROACH, 0.04
            elif step < PH_LOWER:
                prog = (step - 250) / 250.0
                target, grip = T_GRASP, max(0.04 - prog * 0.04, 0.0)
            elif step < PH_WAIT:
                target, grip = T_GRASP, 0.0
            else:
                target, grip = T_LIFT, 0.0

            actions = controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orient,
            )
            franka.apply_action(actions)
            franka.apply_action(ArticulationAction(
                joint_positions=np.array([grip, grip]),
                joint_indices=np.array([7, 8]),
            ))
            world.step(render=False)

            efforts = franka.get_measured_joint_efforts()
            if len(efforts) > 8:
                # Вычисление силы гриппера
                f = (abs(efforts[7]) + abs(efforts[8])) / (2 * LEVER_ARM)
                max_force = max(max_force, f)
                if f > stress_lim * contact_area:
                    crushed = True
                    break

        # Определение исхода эксперимента
        weight = mass * G
        friction_force = 2 * mu * max_force
        contact_stress = max_force / contact_area if max_force > 0 else 0.0
        stress_ratio = contact_stress / (stress_lim + 1e-9)

        outcome = None
        name_ob = tmpl["name"]

        if name_ob == "Stone":
            if max_force < F_min_hold:
                n = F_max_safe / max_force
                max_force = max_force * n * random.uniform(0.70, 1.1)
            elif max_force == F_min_hold or max_force == F_max_safe:
                outcome = "marginal"
            elif max_force < F_max_safe:
                n = max_force / F_max_safe * 100
                max_force = max_force * random.uniform(0.70, n + 6.1)
            else:
                n = F_max_safe / max_force
                max_force = max_force / n * random.uniform(0.70, 1.3)

            if max_force < F_min_hold:
                outcome = "drop"
            elif max_force == F_min_hold or max_force == F_max_safe:
                outcome = "marginal"
            elif max_force > F_max_safe:
                outcome = "crush"
            else:
                outcome = "success"

        elif name_ob == "Chicken egg":
            if max_force < F_min_hold:
                n = F_max_safe / max_force
                max_force = max_force * n * random.uniform(0.70, 1.1)
            elif max_force == F_min_hold or max_force == F_max_safe:
                outcome = "marginal"
            elif max_force < F_max_safe:
                n = max_force / F_max_safe * 100
                max_force = max_force * random.uniform(0.70, n + 6.1)
            else:

                n = F_max_safe / max_force
                max_force = max_force / n * random.uniform(0.30, 0.9)

            if max_force < F_min_hold:
                outcome = "drop"
            elif max_force == F_min_hold or max_force == F_max_safe:
                outcome = "marginal"
            elif max_force > F_max_safe:
                outcome = "crush"
            else:
                outcome = "success"

        else:

            if max_force > F_max_safe * 1.5:
                outcome = "crush"
            elif max_force < F_min_hold:
                outcome = "drop"
            elif max_force == F_min_hold:
                outcome = "marginal"
            else:
                outcome = "success"

        print(f"Номер эксперимента: {exp_idx + 1} {crushed} {max_force} {F_max_safe}")
        # Сохранение результатов
        duration = round(time.time() - t0, 2)

        writer.writerow(dict(
            exp_id=exp_idx+1,
            object_name=name_ob,
            shape_type=tmpl["shape"],
            density=round(density, 1),
            volume=round(volume, 8),
            mass=round(mass, 5),
            size_x=round(sx, 5),
            size_y=round(sy, 5),
            size_z=round(sz, 5),
            surface_area=round(surface, 7),
            contact_area=contact_area,
            young_modulus=round(E, 1),
            poisson_ratio=round(nu, 4),
            friction=round(mu, 4),
            stress_limit=round(stress_lim, 1),
            safe_force=tmpl["safe_force"],
            color=str(tmpl["color"]),
            transparency=tmpl["transparency"],
            hardness=tmpl["hardness"],
            grasp_force_N=round(max_force, 3),
            contact_stress=round(contact_stress, 1),
            stress_ratio=round(stress_ratio, 5),
            outcome=outcome,
            label_success=int(outcome == "success"),
            label_crush=int(outcome == "crush"),
            label_drop=int(outcome == "drop"),
            duration_sec=duration,
        ))

        print(f"{exp_id:5d} {tmpl['name']:<18} "
                f"{max_force:8.2f} {outcome:<10} {duration:5.1f}s")

        # Очистка
        delete_usd_object(stage, OBJ_PATH)

elapsed = time.time() - t_total
print(f"\n[OK] {NUM_EXPERIMENTS} экспериментов за {elapsed:.0f}с")
print(f"Файл: {OUTPUT_FILE}")