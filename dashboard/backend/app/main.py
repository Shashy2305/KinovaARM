"""
KinovaARM demo dashboard backend.

Run:
  source /opt/ros/humble/setup.bash
  source ~/workspace/ros2_kortex_ws/install/setup.bash
  source ~/Shashproject/install/setup.bash
  export ROS_DOMAIN_ID=42
  cd ~/Shashproject/dashboard/backend
  uvicorn app.main:app --host 0.0.0.0 --port 8000

Network-accessible by design (see plan) -- don't expose this past the lab
network, there's no auth on it.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import ros_bridge
from .routers import calibration, camera, command, fusion, nodes, status

app = FastAPI(title='KinovaARM Dashboard API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # lab-network demo tool, no auth either way — see module docstring
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(status.router, prefix='/api')
app.include_router(camera.router, prefix='/api')
app.include_router(nodes.router, prefix='/api')
app.include_router(calibration.router, prefix='/api')
app.include_router(command.router, prefix='/api')
app.include_router(fusion.router, prefix='/api')


@app.on_event('startup')
def on_startup():
    ros_bridge.start_bridge()


@app.get('/api/health')
def health():
    return {'ok': True}
