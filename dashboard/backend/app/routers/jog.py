import asyncio
import json

from fastapi import APIRouter, Body, WebSocket, WebSocketDisconnect

from .. import ros_bridge

router = APIRouter()


def _jogger():
    return ros_bridge.get_or_create_jogger()


@router.get('/jog/state')
async def jog_state():
    j = _jogger()
    await asyncio.to_thread(j.tick_idle)
    return await asyncio.to_thread(j.state)


@router.post('/jog/enable')
async def jog_enable(body: dict = Body(...)):
    """Switches the jog on. Needs the exact phrase; every precondition (live joint states, no fault, arm controller idle, table geometry) is checked first."""
    return await asyncio.to_thread(_jogger().enable, body.get('phrase', ''))


@router.post('/jog/disable')
async def jog_disable():
    _jogger().disable()
    return {'ok': True}


@router.post('/jog/stop')
async def jog_stop():
    """Hold position and switch the jog off. (The hardware E-stop remains the real stop.)"""
    await asyncio.to_thread(_jogger().stop, 'stopped by the Stop button')
    return {'ok': True}


@router.post('/jog/gripper')
async def jog_gripper(body: dict = Body(...)):
    """{'action': 'open' | 'close' | 'set', 'position': 0..0.695}. Needs the jog to be ON."""
    return await asyncio.to_thread(_jogger().gripper, body.get('action'), body.get('position'))


@router.websocket('/ws/jog')
async def ws_jog(websocket: WebSocket):
    """The browser sends one JSON command per tick while a button/stick is held (see Jogger.command) and gets the state back. When the socket closes the arm
    simply stops: nothing is sent, and every command only asks for 0.2 s of motion."""
    await websocket.accept()
    j = _jogger()
    try:
        while True:
            try:
                cmd = json.loads(await websocket.receive_text())
            except ValueError:
                await websocket.send_json({'blocked': 'bad json'})
                continue
            await websocket.send_json(await asyncio.to_thread(j.command, cmd))
    except WebSocketDisconnect:
        pass
