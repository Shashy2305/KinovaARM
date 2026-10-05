import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .. import process_manager, ros_bridge

router = APIRouter()


class AudioControl(BaseModel):
    action: Literal['start', 'stop', 'cancel', 'arm', 'disarm']


@router.post('/audio/control')
def audio_control(req: AudioControl):
    status = process_manager.manager.all_status().get('audio_node', {}).get('status')
    if status not in ('running', 'running_external'):
        raise HTTPException(409, 'The voice input node is not running — start it in Node Control first.')
    ros_bridge.get_bridge().publish_audio_control(req.action)
    return {'ok': True}


@router.websocket('/ws/audio')
async def ws_audio(websocket: WebSocket):
    await websocket.accept()
    bridge = ros_bridge.get_bridge()
    try:
        while True:
            await websocket.send_json(bridge.get_audio_state())
            await asyncio.sleep(0.1)
    except (WebSocketDisconnect, RuntimeError):
        pass
