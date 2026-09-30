import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .. import config, ros_bridge

router = APIRouter()

BOUNDARY = 'frame'


async def _mjpeg_generator(camera_name):
    bridge = ros_bridge.get_bridge()
    while True:
        jpeg, stale = bridge.get_jpeg(camera_name)
        if jpeg is not None and not stale:
            yield (
                b'--' + BOUNDARY.encode() + b'\r\n'
                b'Content-Type: image/jpeg\r\n'
                b'Content-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n'
                + jpeg + b'\r\n'
            )
        await asyncio.sleep(0.1)  # ~10fps is plenty for a demo view


@router.get('/camera/{camera_name}/stream')
async def camera_stream(camera_name: str):
    if camera_name not in config.CAMERA_TOPICS:
        raise HTTPException(404, f'Unknown camera {camera_name!r}')
    return StreamingResponse(
        _mjpeg_generator(camera_name),
        media_type=f'multipart/x-mixed-replace; boundary={BOUNDARY}',
    )


@router.get('/camera/{camera_name}/status')
def camera_status(camera_name: str):
    if camera_name not in config.CAMERA_TOPICS:
        raise HTTPException(404, f'Unknown camera {camera_name!r}')
    bridge = ros_bridge.get_bridge()
    jpeg, stale = bridge.get_jpeg(camera_name)
    return {'has_frame': jpeg is not None, 'stale': stale}
