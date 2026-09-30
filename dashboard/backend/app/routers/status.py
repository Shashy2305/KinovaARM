import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import process_manager, ros_bridge

router = APIRouter()


@router.get('/status')
def get_status():
    bridge = ros_bridge.get_bridge()
    return {
        'ros': bridge.get_status(),
        'processes': process_manager.manager.all_status(),
    }


@router.websocket('/ws/status')
async def ws_status(websocket: WebSocket):
    await websocket.accept()
    bridge = ros_bridge.get_bridge()
    try:
        while True:
            await websocket.send_json({
                'ros': bridge.get_status(),
                'processes': process_manager.manager.all_status(),
            })
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
