"""Module containing high level interface for loading environments."""

import uuid

from holoocean.environments import HoloOceanEnvironment
from holoocean.packagemanager import (
    get_scenario,
    get_binary_path_for_scenario,
    get_package_config_for_scenario,
    get_binary_path_for_package,
)
from holoocean.exceptions import HoloOceanException

from holoocean.util import get_holoocean_path, human_readable_size
import os
from shutil import rmtree as shutilrmtree


class GL_VERSION:
    """OpenGL Version enum.

    Attributes:
        OPENGL3 (:obj:`int`): The value for OpenGL3.
        OPENGL4 (:obj:`int`): The value for OpenGL4.
    """

    OPENGL4 = 4
    OPENGL3 = 3


def make(
    scenario_name="",
    scenario_cfg=None,
    gl_version=GL_VERSION.OPENGL4,
    window_res=None,
    verbose=False,
    show_viewport=True,
    ticks_per_sec=None,
    frames_per_sec=None,
    copy_state=True,
    start_world=True,
):
    """Creates a HoloOcean environment

    Args:
        world_name (:obj:`str`):
            The name of the world to load as an environment. Must match the name of a world in an
            installed package.

        scenario_cfg (:obj:`dict`): Dictionary containing scenario configuration, instead of loading a scenario
            from the installed packages. Dictionary should match the format of the JSON configuration files

        gl_version (:obj:`int`, optional):
            The OpenGL version to use (Linux only). Defaults to GL_VERSION.OPENGL4.

        window_res ((:obj:`int`, :obj:`int`), optional):
            The (height, width) to load the engine window at. Overrides the (optional) resolution in the
            scenario config file

        verbose (:obj:`bool`, optional):
            Whether to run in verbose mode. Defaults to False.

        show_viewport (:obj:`bool`, optional):
            If the viewport window should be shown on-screen (Linux only). Defaults to True

        ticks_per_sec (:obj:`int`, optional):
            The number of frame ticks per unreal seconds. This will override whatever is
            in the configuration json. Defaults to 30.

        frames_per_sec (:obj:`int` or :obj:`bool`, optional):
            The max number of frames ticks per real seconds. This will override whatever is
            in the configuration json. If True, will match ticks_per_sec. If False, will not be
            turned on. If an integer, will set to that value. Defaults to True.

        copy_state (:obj:`bool`, optional):
            If the state should be copied or passed as a reference when returned. Defaults to True

        start_world (:obj:`bool`, optional):
            If the world should be started immediately. Defaults to True
            Set to false if you are running the environment in the editor as play as standalone.

    Returns:
        :class:`~holoocean.environments.HoloOceanEnvironment`: A holoocean environment instantiated
            with all the settings necessary for the specified world, and other supplied arguments.

    """

    param_dict = dict()
    binary_path = None

    if start_world:
        if scenario_name != "":
            scenario = get_scenario(scenario_name)
            binary_path = get_binary_path_for_scenario(scenario_name)
        elif scenario_cfg is not None:
            scenario = scenario_cfg
            binary_path = get_binary_path_for_package(scenario["package_name"])
        else:
            raise HoloOceanException(
                "You must specify scenario_name or scenario_config"
            )
        # Get pre-start steps
        package_config = get_package_config_for_scenario(scenario)
        world = [
            world
            for world in package_config["worlds"]
            if world["name"] == scenario["world"]
        ][0]
        param_dict["pre_start_steps"] = world["pre_start_steps"]
        if "env_min" not in scenario and "env_min" in world:
            scenario["env_min"] = world["env_min"]
            scenario["env_max"] = world["env_max"]
        param_dict["ticks_per_sec"] = ticks_per_sec
        param_dict["frames_per_sec"] = frames_per_sec
        param_dict["scenario"] = scenario
        param_dict["binary_path"] = binary_path
        param_dict["start_world"] = True
        param_dict["uuid"] = str(uuid.uuid4())
        param_dict["gl_version"] = gl_version
        param_dict["verbose"] = verbose
        param_dict["show_viewport"] = show_viewport
        param_dict["copy_state"] = copy_state
        if window_res is not None:
            param_dict["window_size"] = window_res
    else:
        scenario = scenario_cfg
        param_dict["scenario"] = scenario
        param_dict["start_world"] = False
        param_dict["gl_version"] = gl_version
        param_dict["verbose"] = verbose
        param_dict["show_viewport"] = show_viewport
        param_dict["copy_state"] = copy_state

        if "ticks_per_sec" in scenario_cfg:
            ticks_per_sec = scenario_cfg["ticks_per_sec"]

        if ticks_per_sec is None:
            input(
                "Ticks per second not specified, using default of 30. Press enter to continue..."
            )
            ticks_per_sec = 30
        param_dict["ticks_per_sec"] = ticks_per_sec

        if "frames_per_sec" in scenario_cfg:
            frames_per_sec = scenario_cfg["frames_per_sec"]
        if frames_per_sec is None:
            input(
                "Frames per second not specified, using default of 30. Press enter to continue..."
            )
            frames_per_sec = 30
        param_dict["frames_per_sec"] = frames_per_sec
        param_dict["set_fps"] = True if frames_per_sec is not False else None
        param_dict["set_tps"] = True if ticks_per_sec is not None else None

    return HoloOceanEnvironment(**param_dict)


def delete_all_octrees():
    """Deletes the octree from the world.
    If this fails, it will raise a HoloOceanException.
    If it succeeds, it will print the amount of space freed up.
    """
    holoocean_path = get_holoocean_path()
    os.chdir(holoocean_path)
    if not os.path.exists("worlds"):
        raise HoloOceanException("No worlds directory found in HoloOcean path.")
    os.chdir("worlds")
    if not os.path.exists("Ocean"):
        raise HoloOceanException("Ocean not found.")
    os.chdir("Ocean")
    os_type = "Windows" if os.name == "nt" else "Linux"
    if not os.path.exists(os_type):
        raise HoloOceanException("No {} directory found.".format(os_type))
    os.chdir(os_type)
    if not os.path.exists("Holodeck"):
        raise HoloOceanException("No Holodeck directory found.")
    os.chdir("Holodeck")
    if not os.path.exists("Octrees"):
        raise HoloOceanException("No Octrees directory found.")
    os.chdir("Octrees")
    for world_name in os.listdir():
        print("Deleting octree for world: ", world_name)

        def getSize():
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(world_name):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            return total_size

        size_freed = getSize()
        shutilrmtree(world_name)
        size_freed_str = human_readable_size(size_freed)
        print("Freed up ", size_freed_str)


def delete_world_octrees(world_name):
    """Deletes the octree from the specified world.
    If this fails, it will raise a HoloOceanException.
    If it succeeds, it will print the amount of space freed up.
    """
    holoocean_path = get_holoocean_path()
    os.chdir(holoocean_path)
    if not os.path.exists("worlds"):
        raise HoloOceanException("No worlds directory found in HoloOcean path.")
    os.chdir("worlds")
    if not os.path.exists("Ocean"):
        raise HoloOceanException("Ocean not found.")
    os.chdir("Ocean")
    os_type = "Windows" if os.name == "nt" else "Linux"
    if not os.path.exists(os_type):
        raise HoloOceanException("No {} directory found.".format(os_type))
    os.chdir(os_type)
    if not os.path.exists("Holodeck"):
        raise HoloOceanException("No Holodeck directory found.")
    os.chdir("Holodeck")
    if not os.path.exists("Octrees"):
        raise HoloOceanException("No Octrees directory found.")
    os.chdir("Octrees")
    if not os.path.exists(world_name):
        raise HoloOceanException("No octree found for world: {}".format(world_name))
    else:
        print("Deleting octree for world: ", world_name)

        def getSize():
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(world_name):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            return total_size

        size_freed = getSize()
        shutilrmtree(world_name)
        size_freed_str = human_readable_size(size_freed)
        print("Freed up ", size_freed_str)
