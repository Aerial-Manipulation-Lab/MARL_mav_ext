import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

yellow = (1.0, 1.0, 0.0)
red = (1.0, 0.0, 0.0)
green = (0.0, 1.0, 0.0)
blue = (0.0, 0.0, 1.0)


FORCE_MARKER_Z_CFG = VisualizationMarkersCfg(
    markers={
        "arrow_1": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(0.1, 0.1, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=yellow),
        ),
    }
)

ACC_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow_1": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(0.1, 0.1, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=green),
        ),
    }
)

TORQUE_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "frame_1": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.5, 0.5, 0.5),
        ),
        "frame_2": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.5, 0.5, 0.5),
        ),
        "frame_3": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.5, 0.5, 0.5),
        ),
    }
)

GOAL_POS_MARKER_CFG = VisualizationMarkersCfg(
markers={
    "goal_pos_marker": sim_utils.SphereCfg(
        radius=0.05,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    ),
    "current_pos_marker": sim_utils.SphereCfg(
        radius=0.05,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    ),
}
)