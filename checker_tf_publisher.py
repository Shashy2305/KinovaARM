#!/usr/bin/env python3
import rclpy, cv2, numpy as np, tf2_ros
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

CX,CY,SM = 7,10,0.015
SUBPX = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
OBJ = np.zeros((CX*CY,3),np.float32)
OBJ[:,:2] = np.mgrid[0:CX,0:CY].T.reshape(-1,2)*SM

class CheckerTFPublisher(Node):
    def __init__(self):
        super().__init__("checker_tf_publisher")
        self.bridge=CvBridge(); self.K=None; self.dist=None; self.last_t=None
        self.broadcaster=tf2_ros.TransformBroadcaster(self)
        self.create_subscription(CameraInfo,"/global_camera/color/camera_info",self._info,10)
        self.create_subscription(Image,"/global_camera/color/image_raw",self._img,10)
        self.create_timer(0.05,self._timer)
        self.get_logger().info("Checker TF publisher ready")
    def _timer(self):
        if self.last_t:
            self.last_t.header.stamp=self.get_clock().now().to_msg()
            self.broadcaster.sendTransform(self.last_t)
    def _info(self,msg):
        if self.K is None:
            self.K=np.array(msg.k).reshape(3,3); self.dist=np.array(msg.d)
            self.get_logger().info(f"Intrinsics ok fx={self.K[0,0]:.1f}")
    def _img(self,msg):
        if self.K is None: return
        gray=cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg,"bgr8"),cv2.COLOR_BGR2GRAY)
        found,c=cv2.findChessboardCorners(gray,(CX,CY),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_NORMALIZE_IMAGE|cv2.CALIB_CB_FAST_CHECK)
        if not found: return
        c=cv2.cornerSubPix(gray,c,(11,11),(-1,-1),SUBPX)
        ok,rvec,tvec=cv2.solvePnP(OBJ,c,self.K,self.dist,flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: return
        R,_=cv2.Rodrigues(rvec); tv=tvec.flatten()
        qx,qy,qz,qw=self._q(R)
        t=TransformStamped()
        t.header.stamp=self.get_clock().now().to_msg()
        t.header.frame_id="global_camera_link"; t.child_frame_id="checkerboard_frame"
        t.transform.translation.x=float(tv[0]); t.transform.translation.y=float(tv[1]); t.transform.translation.z=float(tv[2])
        t.transform.rotation.x=qx; t.transform.rotation.y=qy; t.transform.rotation.z=qz; t.transform.rotation.w=qw
        self.last_t=t
        self.get_logger().info(f"Board detected dist={np.linalg.norm(tv)*100:.1f}cm",throttle_duration_sec=1.0)
    def _q(self,R):
        tr=R[0,0]+R[1,1]+R[2,2]
        if tr>0:
            s=0.5/np.sqrt(tr+1.0); return (R[2,1]-R[1,2])*s,(R[0,2]-R[2,0])*s,(R[1,0]-R[0,1])*s,0.25/s
        elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
            s=2.0*np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2]); return 0.25*s,(R[0,1]+R[1,0])/s,(R[0,2]+R[2,0])/s,(R[2,1]-R[1,2])/s
        elif R[1,1]>R[2,2]:
            s=2.0*np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2]); return (R[0,1]+R[1,0])/s,0.25*s,(R[1,2]+R[2,1])/s,(R[0,2]-R[2,0])/s
        else:
            s=2.0*np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1]); return (R[0,2]+R[2,0])/s,(R[1,2]+R[2,1])/s,0.25*s,(R[1,0]-R[0,1])/s

def main():
    rclpy.init(); rclpy.spin(CheckerTFPublisher()); rclpy.shutdown()

if __name__ == "__main__":
    main()
