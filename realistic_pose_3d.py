"""
Realistic Image → 3D Posed Human Mesh
========================================
Uses MediaPipe for pose detection + SMPL body model for realistic mesh.
Falls back to high-quality sculpted mesh if SMPL unavailable.

INSTALL:
    pip install mediapipe opencv-python numpy

OPTIONAL (for true SMPL mesh):
    pip install smplx torch
    Download SMPL model from https://smpl.is.tue.mpg.de/
    Place SMPL_NEUTRAL.pkl in smpl_models/

USAGE:
    python realistic_pose_3d.py --image person.jpg

OUTPUT:
    person_posed.obj  ← realistic 3D human in exact pose from photo
    person_posed.mtl  ← material
    person_preview.jpg ← skeleton overlay
"""

import cv2, numpy as np, mediapipe as mp, os, math, argparse, sys

def detect_joints(image_path):
    img = cv2.imread(image_path)
    if img is None: sys.exit(f"Cannot read: {image_path}")
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=2,
                      min_detection_confidence=0.5) as pose:
        result = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not result.pose_world_landmarks:
        sys.exit("No person detected.")
    LM  = mp_pose.PoseLandmark
    wlm = result.pose_world_landmarks.landmark
    def w(n):
        l = wlm[LM[n].value]
        return np.array([l.x, -l.y, -l.z], np.float64)
    J = {k: w(v) for k,v in {
        "nose":"NOSE","left_ear":"LEFT_EAR","right_ear":"RIGHT_EAR",
        "left_shoulder":"LEFT_SHOULDER","right_shoulder":"RIGHT_SHOULDER",
        "left_elbow":"LEFT_ELBOW","right_elbow":"RIGHT_ELBOW",
        "left_wrist":"LEFT_WRIST","right_wrist":"RIGHT_WRIST",
        "left_hand":"LEFT_INDEX","right_hand":"RIGHT_INDEX",
        "left_hip":"LEFT_HIP","right_hip":"RIGHT_HIP",
        "left_knee":"LEFT_KNEE","right_knee":"RIGHT_KNEE",
        "left_ankle":"LEFT_ANKLE","right_ankle":"RIGHT_ANKLE",
        "left_foot":"LEFT_FOOT_INDEX","right_foot":"RIGHT_FOOT_INDEX",
    }.items()}
    # save preview
    mp.solutions.drawing_utils.draw_landmarks(
        img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    prev = os.path.splitext(image_path)[0]+"_preview.jpg"
    cv2.imwrite(prev, img)
    print(f"  Preview -> {prev}")
    return J


def try_smpl_posed(joints, smpl_dir="smpl_models/"):
    """Try to load SMPL and pose it with detected joint angles."""
    try:
        import torch, smplx
        # find model file
        candidates = [
            "SMPL_NEUTRAL.pkl","basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
            "basicmodel_neutral_lbs_10_207_0_v1.0.0.pkl","SMPL_MALE.pkl",
        ]
        found = None
        for c in candidates:
            p = os.path.join(smpl_dir, c)
            if os.path.exists(p): found=p; break
        if not found: return None, None

        model = smplx.create(smpl_dir, model_type='smpl',
                             gender='neutral', num_betas=10, ext='pkl')

        pose_params = joints_to_smpl_pose(joints)
        betas       = torch.zeros(1, 10)
        body_pose   = torch.tensor(pose_params[3:], dtype=torch.float32).unsqueeze(0)
        global_orient = torch.tensor(pose_params[:3], dtype=torch.float32).unsqueeze(0)

        output = model(betas=betas, body_pose=body_pose,
                       global_orient=global_orient, return_verts=True)
        verts = output.vertices[0].detach().cpu().numpy()
        faces = model.faces.astype(np.int32)
        print(f"  SMPL mesh: {len(verts)} verts, {len(faces)} faces")
        return verts, faces
    except Exception as e:
        print(f"  SMPL not available ({e}) → using sculpted mesh")
        return None, None

def joints_to_smpl_pose(J):
    """Convert MediaPipe joints to SMPL 72-dim pose vector (axis-angle)."""
    params = np.zeros(72, dtype=np.float32)

    def bone_axis_angle(parent, child, rest=None):
        if rest is None: rest = np.array([0,-1,0])
        d = J[child] - J[parent]
        n = np.linalg.norm(d)
        if n < 1e-8: return np.zeros(3)
        d /= n
        rest = rest / np.linalg.norm(rest)
        axis = np.cross(rest, d)
        al   = np.linalg.norm(axis)
        if al < 1e-8: return np.zeros(3)
        axis /= al
        angle = math.acos(np.clip(np.dot(rest,d),-1,1))
        return axis * angle

    

    mid_hip = (J["left_hip"]+J["right_hip"])/2
    mid_sho = (J["left_shoulder"]+J["right_shoulder"])/2

    spine_d = mid_sho - mid_hip
    params[0:3] = bone_axis_angle("left_hip","left_shoulder",
                                   rest=np.array([0,1,0])) * 0.1

    params[3:6]   = bone_axis_angle("left_hip","left_knee")
    params[6:9]   = bone_axis_angle("right_hip","right_knee")
    params[12:15] = bone_axis_angle("left_knee","left_ankle")
    params[15:18] = bone_axis_angle("right_knee","right_ankle")
    params[21:24] = bone_axis_angle("left_ankle","left_foot",
                                     rest=np.array([0,0,1]))
    params[24:27] = bone_axis_angle("right_ankle","right_foot",
                                     rest=np.array([0,0,1]))
    params[48:51] = bone_axis_angle("left_shoulder","left_elbow")
    params[51:54] = bone_axis_angle("right_shoulder","right_elbow")
    params[54:57] = bone_axis_angle("left_elbow","left_wrist")
    params[57:60] = bone_axis_angle("right_elbow","right_wrist")
    params[60:63] = bone_axis_angle("left_wrist","left_hand")
    params[63:66] = bone_axis_angle("right_wrist","right_hand")

    return params


def norm(v):
    n=np.linalg.norm(v); return v/n if n>1e-8 else v

def local_frame(axis):
    axis=norm(np.array(axis,float))
    ref=np.array([0,1,0]) if abs(axis[1])<0.9 else np.array([1,0,0])
    u=norm(np.cross(ref,axis)); v=np.cross(axis,u)
    return u,v

def muscle_cylinder(p0, p1, r_base, segs=16, bulge=1.15, taper=0.75):
    """
    Limb segment with muscle belly bulge in the middle and taper at joints.
    bulge: max radius multiplier at midpoint
    taper: radius fraction at endpoints
    """
    p0,p1=np.array(p0,float),np.array(p1,float)
    d=p1-p0; L=np.linalg.norm(d)
    if L<1e-8: return np.zeros((0,3)),np.zeros((0,3),int)
    u,v=local_frame(d)
    rings=20  # longitudinal resolution
    th=np.linspace(0,2*math.pi,segs,endpoint=False)
    circle=np.array([math.cos(a)*u+math.sin(a)*v for a in th])
    all_verts=[]
    for i in range(rings+1):
        t=i/rings
        # muscle profile: tapered ends, bulge in upper third
        if t<0.15:
            r=r_base*(taper + (1-taper)*(t/0.15))
        elif t<0.45:
            r=r_base*(1.0 + (bulge-1.0)*((t-0.15)/0.30))
        elif t<0.70:
            r=r_base*bulge
        else:
            r=r_base*(bulge - (bulge-taper)*((t-0.70)/0.30))
        pt=p0+d*t
        all_verts.append(pt + circle*r)
    verts=np.vstack(all_verts)
    faces=[]
    for i in range(rings):
        for j in range(segs):
            k=(j+1)%segs
            a=i*segs+j; b=i*segs+k; c=(i+1)*segs+j; dd=(i+1)*segs+k
            faces+=[[a,b,dd],[a,dd,c]]
    # caps
    bc=len(verts); verts=np.vstack([verts,[p0]])
    tc=len(verts); verts=np.vstack([verts,[p1]])
    for j in range(segs):
        k=(j+1)%segs
        faces.append([bc,k,j])
        faces.append([tc,rings*segs+j,rings*segs+k])
    return verts,np.array(faces,int)

def shaped_torso(mid_hip, mid_sho, left_hip, right_hip,
                  left_sho, right_sho, gender='female'):
    """
    Build a realistic torso mesh with chest/waist/hip shaping.
    Uses 8 cross-section rings from hip to shoulder.
    """
    hip_w  = np.linalg.norm(left_hip-right_hip)
    sho_w  = np.linalg.norm(left_sho-right_sho)
    spine  = mid_sho - mid_hip
    L      = np.linalg.norm(spine)
    if L<1e-8: return np.zeros((0,3)),np.zeros((0,3),int)

    u, v   = local_frame(spine)
    # perpendicular to spine in frontal plane
    right_dir = norm(right_sho - left_sho)
    depth_dir = norm(np.cross(spine/L, right_dir))

    segs   = 18
    rings  = 12
    th     = np.linspace(0,2*math.pi,segs,endpoint=False)

    # Width & depth profiles along spine (0=hip, 1=shoulder)
    # female: wider hips, narrow waist, moderate chest
    if gender=='female':
        w_profile = [1.00,0.98,0.90,0.80,0.72,0.74,0.82,0.88,0.92,0.95,0.97,1.00]
        d_profile = [0.55,0.53,0.48,0.43,0.40,0.42,0.46,0.50,0.52,0.53,0.52,0.50]
    else:
        w_profile = [0.88,0.87,0.84,0.81,0.80,0.82,0.86,0.91,0.95,0.98,1.00,1.00]
        d_profile = [0.52,0.51,0.50,0.49,0.48,0.49,0.51,0.53,0.55,0.56,0.55,0.53]

    all_v=[]
    for i in range(rings):
        t      = i/(rings-1)
        pt     = mid_hip + spine*t
        wi     = hip_w * w_profile[i] * 0.52
        di     = hip_w * d_profile[i]
        ring   = []
        for a in th:
            # Superellipse cross section (more box-like than circle)
            n=2.4
            cx=math.copysign(abs(math.cos(a))**(2/n),math.cos(a))
            cy=math.copysign(abs(math.sin(a))**(2/n),math.sin(a))
            ring.append(pt + right_dir*cx*wi + depth_dir*cy*di)
        all_v.append(ring)

    verts = np.array(all_v).reshape(-1,3)
    faces=[]
    for i in range(rings-1):
        for j in range(segs):
            k=(j+1)%segs
            a=i*segs+j; b=i*segs+k; c=(i+1)*segs+j; d=(i+1)*segs+k
            faces+=[[a,b,d],[a,d,c]]
    return verts,np.array(faces,int)

def realistic_head(center, ear_l, ear_r, nose, sho_w):
    """Build a realistic head shape: slightly flattened sphere with facial volume."""
    R    = sho_w*0.40
    segs = 20
    stacks=16
    verts=[]
    # Oval head: wider at ears, slightly flat front-back
    rx,ry,rz = R*1.0, R*1.15, R*0.90
    for i in range(stacks+1):
        lat=math.pi*i/stacks-math.pi/2
        for j in range(segs):
            lon=2*math.pi*j/segs
            verts.append([
                center[0]+rx*math.cos(lat)*math.cos(lon),
                center[1]+ry*math.sin(lat),
                center[2]+rz*math.cos(lat)*math.sin(lon),
            ])
    verts=np.array(verts)
    faces=[]
    for i in range(stacks):
        for j in range(segs):
            a=i*segs+j; b=i*segs+(j+1)%segs
            c=(i+1)*segs+j; d=(i+1)*segs+(j+1)%segs
            faces+=[[a,b,d],[a,d,c]]
    return verts,np.array(faces,int)

def realistic_foot(ankle, foot_tip, shin_r):
    """Build a realistic foot shape."""
    fd    = norm(foot_tip - ankle)
    # up direction (perpendicular to foot, pointing up)
    right = norm(np.cross(fd, np.array([0,1,0])))
    up    = norm(np.cross(right, fd))
    flen  = shin_r*3.2
    fw    = shin_r*0.85
    fh    = shin_r*0.55

    # 8 cross sections from heel to toe
    segs=12; rings=8
    verts=[]
    for i in range(rings):
        t=i/(rings-1)
        # taper at toe
        w = fw*(1.0-t*0.45)
        h = fh*(1.0-t*0.30)
        ctr = ankle + fd*flen*t - up*shin_r*0.4
        th=np.linspace(0,2*math.pi,segs,endpoint=False)
        for a in th:
            verts.append(ctr + right*math.cos(a)*w + up*math.sin(a)*h)

    verts=np.array(verts)
    faces=[]
    for i in range(rings-1):
        for j in range(segs):
            k=(j+1)%segs
            a=i*segs+j; b=i*segs+k; c=(i+1)*segs+j; d=(i+1)*segs+k
            faces+=[[a,b,d],[a,d,c]]
    return verts,np.array(faces,int)

def realistic_hand(wrist, hand_tip, forearm_r):
    """Build a realistic hand shape."""
    fd    = norm(hand_tip - wrist)
    right = norm(np.cross(fd, np.array([0,1,0])+np.array([0,0,0.01])))
    up    = norm(np.cross(right, fd))
    hlen  = forearm_r*2.4
    hw    = forearm_r*0.9
    hh    = forearm_r*0.35

    segs=10; rings=6
    verts=[]
    for i in range(rings):
        t=i/(rings-1)
        w=hw*(1.0-t*0.3); h=hh*(1.0-t*0.2)
        ctr=wrist+fd*hlen*t
        th=np.linspace(0,2*math.pi,segs,endpoint=False)
        for a in th:
            verts.append(ctr + right*math.cos(a)*w + up*math.sin(a)*h)

    verts=np.array(verts)
    faces=[]
    for i in range(rings-1):
        for j in range(segs):
            k=(j+1)%segs
            a=i*segs+j; b=i*segs+k; c=(i+1)*segs+j; d=(i+1)*segs+k
            faces+=[[a,b,d],[a,d,c]]
    return verts,np.array(faces,int)

def joint_sphere(center, r, segs=10):
    """Smooth sphere for joint transitions."""
    stacks=segs; verts=[]
    for i in range(stacks+1):
        lat=math.pi*i/stacks-math.pi/2
        for j in range(segs):
            lon=2*math.pi*j/segs
            verts.append([
                center[0]+r*math.cos(lat)*math.cos(lon),
                center[1]+r*math.sin(lat),
                center[2]+r*math.cos(lat)*math.sin(lon),
            ])
    verts=np.array(verts); faces=[]
    for i in range(stacks):
        for j in range(segs):
            a=i*segs+j; b=i*segs+(j+1)%segs
            c=(i+1)*segs+j; d=(i+1)*segs+(j+1)%segs
            faces+=[[a,b,d],[a,d,c]]
    return verts,np.array(faces,int)

def build_realistic_mesh(J):
    all_v=[]; all_f=[]; off=0
    def add(verts,faces):
        nonlocal off
        if len(verts)==0: return
        all_v.append(verts); all_f.append(faces+off); off+=len(verts)

    mid_sho=(J["left_shoulder"]+J["right_shoulder"])/2
    mid_hip=(J["left_hip"]+J["right_hip"])/2
    sho_w  =np.linalg.norm(J["left_shoulder"]-J["right_shoulder"])
    hip_w  =np.linalg.norm(J["left_hip"]-J["right_hip"])

    # radii
    r_ua = sho_w*0.115   # upper arm
    r_fa = sho_w*0.092   # forearm
    r_th = hip_w*0.265   # thigh
    r_sh = hip_w*0.185   # shin
    r_ank= hip_w*0.115   # ankle

    # ── HEAD ──
    ear_mid=(J["left_ear"]+J["right_ear"])/2
    head_ctr=(ear_mid*0.6+J["nose"]*0.4)+np.array([0,sho_w*0.08,0])
    add(*realistic_head(head_ctr,J["left_ear"],J["right_ear"],J["nose"],sho_w))

    # ── NECK ──
    neck_bot=mid_sho+np.array([0,sho_w*0.02,0])
    neck_top=head_ctr-np.array([0,sho_w*0.38,0])
    add(*muscle_cylinder(neck_bot,neck_top,sho_w*0.085,segs=12,bulge=1.05,taper=0.90))

    # ── TORSO ──
    add(*shaped_torso(mid_hip,mid_sho,
                       J["left_hip"],J["right_hip"],
                       J["left_shoulder"],J["right_shoulder"],
                       gender='female'))

    # ── ARMS ──
    for side in ("left","right"):
        sho   =J[f"{side}_shoulder"]
        elbow =J[f"{side}_elbow"]
        wrist =J[f"{side}_wrist"]
        hand  =J[f"{side}_hand"]

        # shoulder cap
        add(*joint_sphere(sho, r_ua*1.05, segs=10))
        # upper arm — bicep bulge
        add(*muscle_cylinder(sho,elbow,r_ua,segs=14,bulge=1.22,taper=0.78))
        # elbow cap
        add(*joint_sphere(elbow, r_ua*0.88, segs=8))
        # forearm — slight taper
        add(*muscle_cylinder(elbow,wrist,r_fa,segs=12,bulge=1.12,taper=0.72))
        # hand
        add(*realistic_hand(wrist,hand,r_fa))

    # ── LEGS ──
    for side in ("left","right"):
        hip   =J[f"{side}_hip"]
        knee  =J[f"{side}_knee"]
        ankle =J[f"{side}_ankle"]
        foot  =J[f"{side}_foot"]

        add(*joint_sphere(hip, r_th*0.80, segs=10))
        # thigh — quad bulge
        add(*muscle_cylinder(hip,knee,r_th,segs=16,bulge=1.18,taper=0.75))
        # knee cap
        add(*joint_sphere(knee, r_th*0.68, segs=8))
        # shin — calf bulge
        add(*muscle_cylinder(knee,ankle,r_sh,segs=14,bulge=1.20,taper=0.60))
        # ankle
        add(*joint_sphere(ankle, r_ank*1.1, segs=8))
        # foot
        add(*realistic_foot(ankle,foot,r_sh))

    verts=np.concatenate(all_v,0).astype(np.float32)
    faces=np.concatenate(all_f,0).astype(np.int32)
    print(f"  Mesh: {len(verts)} verts, {len(faces)} faces")
    return verts,faces

# ══════════════════════════════════════════════════════════════════════════════
# 4. EXPORT
# ══════════════════════════════════════════════════════════════════════════════
def sample_colors(image_path):
    img=cv2.imread(image_path)
    if img is None: return {"skin":(0.78,0.57,0.43)}
    h,w=img.shape[:2]
    def crop_mean(y1,y2,x1,x2):
        c=img[int(h*y1):int(h*y2),int(w*x1):int(w*x2)]
        if c.size==0: return (0.5,0.5,0.5)
        m=c.reshape(-1,3).mean(0)
        return (float(m[2]/255),float(m[1]/255),float(m[0]/255))
    return {
        "skin":  crop_mean(0.05,0.18,0.38,0.62),
        "shirt": crop_mean(0.18,0.40,0.30,0.70),
        "pants": crop_mean(0.42,0.65,0.30,0.70),
        "shoes": crop_mean(0.88,1.00,0.30,0.70),
    }

def save_mtl(path, colors):
    with open(path,'w') as f:
        for name,(r,g,b) in colors.items():
            f.write(f"newmtl {name}\nKd {r:.3f} {g:.3f} {b:.3f}\n"
                    f"Ka 0.15 0.10 0.08\nKs 0.06 0.06 0.06\nNs 15\n\n")

def save_obj(verts, faces, obj_path, mtl_name):
    with open(obj_path,'w') as f:
        f.write(f"mtllib {mtl_name}.mtl\nusemtl skin\n")
        for v in verts: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("s 1\n")
        for fc in faces: f.write(f"f {fc[0]+1} {fc[1]+1} {fc[2]+1}\n")
    print(f"  OBJ -> {obj_path}")


def save_blender_script(obj_path, out_dir):
    script=f'''"""
Blender import + smooth script
Run in Blender Scripting tab after importing the OBJ.
"""
import bpy, os
# Import
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.wm.obj_import(filepath=r"{obj_path}")
obj = bpy.context.selected_objects[0]
obj.name = "PosedHuman"
bpy.context.view_layer.objects.active = obj

# Smooth shading
bpy.ops.object.shade_smooth()

# Subdivision surface for smoothness
mod = obj.modifiers.new("Subdiv","SUBSURF")
mod.levels = 2
mod.render_levels = 3

# Corrective smooth
cs = obj.modifiers.new("CorrectiveSmooth","CORRECTIVE_SMOOTH")
cs.factor = 0.5
cs.iterations = 5

print("Done! Human mesh imported and smoothed.")
'''
    p=os.path.join(out_dir,"blender_smooth.py")
    with open(p,'w') as f: f.write(script)
    print(f"  Blender script -> {p}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--image",   required=True)
    ap.add_argument("--output",  default=None)
    ap.add_argument("--smpl",    default="smpl_models/")
    args=ap.parse_args()

    base=os.path.splitext(args.image)[0]
    out =args.output or base+"_posed.obj"
    out_dir=os.path.dirname(out) or "."
    os.makedirs(out_dir,exist_ok=True)
    mtl_base=os.path.splitext(os.path.basename(out))[0]
    mtl_path=os.path.splitext(out)[0]+".mtl"

    print("\n=== REALISTIC IMAGE -> 3D POSED MESH ===")
    print(f"Input:  {args.image}")
    print(f"Output: {out}\n")

    print("[1/3] Detecting pose...")
    joints=detect_joints(args.image)

    print("[2/3] Building realistic 3D mesh...")
    # Try SMPL first
    verts,faces=try_smpl_posed(joints,args.smpl)
    if verts is None:
        verts,faces=build_realistic_mesh(joints)

    print("[3/3] Saving files...")
    colors=sample_colors(args.image)
    save_mtl(mtl_path,colors)
    save_obj(verts,faces,out,mtl_base)
    save_blender_script(os.path.abspath(out),out_dir)

    print(f"\n=== DONE ===")
    print(f"  3D mesh:  {out}")
    print(f"  Material: {mtl_path}")
    print(f"\nFor smoothest result in Blender:")
    print(f"  File > Import > Wavefront OBJ > select {os.path.basename(out)}")
    print(f"  Then run blender_smooth.py in Scripting tab")

if __name__=="__main__":
    main()
