from fastapi import APIRouter
from pydantic import BaseModel

from .. import ros_bridge

router = APIRouter()


class CommandRequest(BaseModel):
    text: str


@router.post('/command')
def send_command(req: CommandRequest):
    bridge = ros_bridge.get_bridge()
    bridge.publish_voice_command(req.text)
    return {'ok': True, 'message': f'Published "{req.text}" to /voice_command.'}
