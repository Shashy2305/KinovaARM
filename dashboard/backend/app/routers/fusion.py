import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import ros_bridge

router = APIRouter()


@router.websocket('/ws/fusion')
async def ws_fusion(websocket: WebSocket):
    """Streams the fused point cloud as raw binary: an interleaved
    float32 [x,y,z,r,g,b, ...] buffer per frame, sent as a WS binary
    message. ~4Hz is plenty for a demo-smooth rotating point cloud without
    saturating the connection -- see ros_bridge.get_fused_points()."""
    await websocket.accept()
    bridge = ros_bridge.get_bridge()
    try:
        while True:
            points = bridge.get_fused_points()
            await websocket.send_bytes(points.tobytes())
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        pass
