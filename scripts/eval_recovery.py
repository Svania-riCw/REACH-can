"""Physical retreat in OSC_POSE coordinates; never restore simulator state here."""
import numpy as np
from scipy.spatial.transform import Rotation


def invert_se3(transform):
    transform=np.asarray(transform,dtype=np.float64)
    result=np.eye(4)
    result[:3,:3]=transform[:3,:3].T
    result[:3,3]=-result[:3,:3]@transform[:3,3]
    return result


def pose_error(current,target):
    return (float(np.linalg.norm(target[:3,3]-current[:3,3])),
            float(Rotation.from_matrix(target[:3,:3]@current[:3,:3].T).magnitude()))


def retreat_command(current,target,controller,gripper):
    """Convert desired inverse SE(3) motion to the world's OSC delta convention.

    Robosuite applies world translations and LEFT-multiplied world rotations;
    simply negating a normalized 7D action is not a generic SE(3) inverse.
    """
    physical=np.r_[target[:3,3]-current[:3,3],
                    Rotation.from_matrix(target[:3,:3]@current[:3,:3].T).as_rotvec()]
    low,high=np.asarray(controller['output_min']),np.asarray(controller['output_max'])
    in_low,in_high=np.asarray(controller['input_min']),np.asarray(controller['input_max'])
    if np.any(high<=low):raise ValueError('Degenerate OSC scaling')
    command=(physical-(high+low)/2)*((in_high-in_low)/(high-low))+(in_high+in_low)/2
    return np.r_[np.clip(command,in_low,in_high),np.clip(gripper,-1,1)]


def failed_prompt(history,horizon):
    if len(history)==0:raise ValueError('A failed prompt requires actually executed actions')
    values=np.asarray(history,dtype=np.float32)[-horizon:]
    if values.ndim!=2 or values.shape[1]!=7:raise ValueError('Expected executed environment actions [T,7]')
    return np.concatenate([values,np.repeat(values[-1:],horizon-len(values),axis=0)])
