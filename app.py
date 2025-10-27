# app.py  -- Improved MVP with mask painting & optional ONNX segmentation
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
from pyembroidery import EmbPattern, STITCH, JUMP, TRIM, COLOR_CHANGE, write_dst
import matplotlib.pyplot as plt
import base64
import uuid
import onnxruntime as ort
from scipy.spatial import KDTree

# === CONFIG ===
ONNX_MODEL_PATH = None  # set to "models/your_model.onnx" to enable ONNX segmentation
THREAD_PALETTE = [
    (255,255,255),(0,0,0),(255,0,128),(255,128,0),(128,0,255),(64,128,192),(255,192,203),(128,128,128)
]
THREAD_TREE = KDTree(THREAD_PALETTE)

app = FastAPI(title="AI Embroidery Digitizer (MVP + Edit + ONNX)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# === Optional ONNX wrapper ===
class ONNXSegmenter:
    def __init__(self, path=None):
        self.sess = None
        if path:
            try:
                self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                print("Loaded ONNX model:", path)
            except Exception as e:
                print("Failed to load ONNX model:", e)
                self.sess = None
    def predict(self, rgb: np.ndarray):
        # Expect rgb in HxWx3 (uint8). Resize to model input (we'll assume 256x256).
        if self.sess is None:
            return None
        inp = self.sess.get_inputs()[0]
        H, W = inp.shape[-2], inp.shape[-1] if len(inp.shape)>=3 else (256,256)
        x = cv2.resize(rgb, (W,H)).astype(np.float32)/255.0
        x = np.transpose(x, (2,0,1))[None, ...]
        try:
            outs = self.sess.run(None, {self.sess.get_inputs()[0].name: x})
        except Exception as e:
            print("ONNX run error:", e)
            return None
        out = outs[0][0]
        if out.ndim==3:
            seg = np.argmax(out, axis=0).astype(np.uint8)
        else:
            seg = out.astype(np.uint8)
        seg = cv2.resize(seg, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        return seg

onnx_seg = ONNXSegmenter(ONNX_MODEL_PATH)

# === Utilities ===
def read_image_to_rgb(file_bytes: bytes, max_dim: int = 1200):
    img = Image.open(BytesIO(file_bytes)).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255,255,255,255))
    img = Image.alpha_composite(bg, img).convert("RGB")
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    return img

def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def kmeans_segmentation(bgr, k=6):
    H,W,_ = bgr.shape
    flat = bgr.reshape(-1,3).astype(np.float32)
    km = KMeans(n_clusters=k, n_init=4, random_state=0)
    labels = km.fit_predict(flat)
    return labels.reshape(H,W)

def clean_mask(mask, min_area=200):
    m = (mask>0).astype(np.uint8)*255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
    num, comps, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[comps==i] = 255
    return out

def smooth_contour(cnt, eps_factor=0.01):
    peri = cv2.arcLength(cnt, True)
    eps = eps_factor * peri
    approx = cv2.approxPolyDP(cnt, eps, True)
    if len(approx) < 6:
        approx = cnt.reshape(-1,1,2)
    return approx

def rotated_scanlines_fill(mask, spacing_px=6, angle_deg=45):
    h,w = mask.shape
    cx, cy = w/2.0, h/2.0
    M = cv2.getRotationMatrix2D((cx,cy), -angle_deg, 1.0)
    rotated = cv2.warpAffine(mask, M, (w,h), flags=cv2.INTER_NEAREST, borderValue=0)
    segments = []
    for y in range(0, h, spacing_px):
        row = rotated[y]
        xs = np.where(row>0)[0]
        if xs.size==0: continue
        groups = np.split(xs, np.where(np.diff(xs)!=1)[0]+1)
        for g in groups:
            if g.size==0: continue
            x1, x2 = int(g[0]), int(g[-1])
            p1 = cv2.transform(np.array([[[x1,y]]], dtype=np.float32), cv2.getRotationMatrix2D((cx,cy), angle_deg,1.0))[0][0]
            p2 = cv2.transform(np.array([[[x2,y]]], dtype=np.float32), cv2.getRotationMatrix2D((cx,cy), angle_deg,1.0))[0][0]
            segments.append((tuple(p1), tuple(p2)))
    return segments

def map_color_to_thread(rgb):
    _, idx = THREAD_TREE.query(rgb)
    return THREAD_PALETTE[int(idx)]

def add_run_from_points(pattern: EmbPattern, pts, scale=1.0):
    if pts is None or len(pts)==0: return
    x0,y0 = int(pts[0][0]*scale), int(pts[0][1]*scale)
    pattern.add_stitch_absolute(JUMP, x0, y0)
    for (x,y) in pts:
        pattern.add_stitch_absolute(STITCH, int(x*scale), int(y*scale))

def add_line_segment(pattern: EmbPattern, p1, p2, scale=1.0):
    x1,y1 = int(p1[0]*scale), int(p1[1]*scale)
    x2,y2 = int(p2[0]*scale), int(p2[1]*scale)
    pattern.add_stitch_absolute(JUMP, x1, y1)
    pattern.add_stitch_absolute(STITCH, x2, y2)

# === Core planner ===
def plan_stitches(bgr, seg, px_per_mm=10.0, fill_spacing_mm=0.5, fill_angle=45.0, mask_override=None):
    scale = (1.0/px_per_mm)*10.0
    pattern = EmbPattern()
    # if mask_override provided, it should be same shape as seg and nonzero where user painted
    if mask_override is not None:
        # apply override by replacing corresponding labels with a unique label
        seg = seg.copy()
        seg[mask_override>0] = seg.max()+1
    labels = np.unique(seg)[:16]
    h,w,_ = bgr.shape
    spacing_px = max(1, int(fill_spacing_mm * px_per_mm))
    for i, lab in enumerate(labels):
        mask = (seg==lab).astype(np.uint8)*255
        mask = clean_mask(mask, min_area=80)
        if mask.max()==0: continue
        color = cv2.mean(bgr, mask=mask)[:3]
        if i>0:
            pattern.add_command(COLOR_CHANGE)
        x,y,ww,hh = cv2.boundingRect(mask)
        # thin region -> satin-like centerline approximated
        if min(ww,hh) <= max(6, int(2.0*px_per_mm)):
            contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours: continue
            cnt = max(contours, key=cv2.contourArea)
            sc = smooth_contour(cnt, eps_factor=0.01).reshape(-1,2)
            # simple centerline: use centroid midpoint method
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
                center = np.array([cx,cy])
                pts = [tuple(((p+center)/2.0).astype(int)) for p in sc]
                add_run_from_points(pattern, pts, scale=scale)
            else:
                add_run_from_points(pattern, sc, scale=scale)
        else:
            # tatami fill using rotated scanlines
            lines = rotated_scanlines_fill(mask, spacing_px=spacing_px, angle_deg=fill_angle)
            for (p1,p2) in lines:
                add_line_segment(pattern, p1, p2, scale=scale)
            contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                sc = smooth_contour(cnt, eps_factor=0.01).reshape(-1,2)
                add_run_from_points(pattern, sc, scale=scale)
    pattern.add_command(TRIM)
    return pattern

# === HTML UI with canvas painter ===
INDEX_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>AI Emb Digitizer - Edit</title>
<style>body{font-family:Arial;max-width:1100px;margin:18px auto} .card{border:1px solid #ddd;padding:18px;border-radius:8px}</style>
</head><body>
<h2>AI Embroidery Digitizer — Upload + Paint Fix + Export</h2>
<div class="card">
<form id="f">
<input id="file" type="file" name="file" accept="image/*" required />
<label>Colors K: <input name="k" value="8" type="number" min="2" max="16"/></label>
<label>px/mm: <input name="pxmm" value="10" step="0.1"/></label>
<label>Fill spacing (mm): <input name="fill" value="0.5" step="0.1"/></label>
<label>Fill angle: <input name="angle" value="45" step="1"/></label>
<label>Use ONNX (if configured server-side): <input name="use_onnx" type="checkbox" /></label>
<button id="previewBtn" type="button">Preview Stitches</button>
<button id="exportBtn" type="button">Auto Digitize → DST</button>
</form>
<div style="margin-top:12px;">
<canvas id="canvas" style="border:1px solid #ccc;max-width:100%"></canvas>
</div>
<div style="margin-top:8px;">
<button id="pencil">Pencil</button><button id="eraser">Eraser</button>
<label>Size: <input id="size" type="range" min="2" max="60" value="20"/></label>
</div>
<div id="preview" style="margin-top:12px"></div>
</div>

<script>
let origImg = null;
let maskCanvas=null, maskCtx=null, cv=null;
const fileEl = document.getElementById('file');
fileEl.addEventListener('change', async ()=>{
  if(!fileEl.files || !fileEl.files[0]) return;
  const f = fileEl.files[0];
  const img = await createImageBitmap(f);
  const canvas = document.getElementById('canvas');
  canvas.width = img.width; canvas.height = img.height;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(img,0,0);
  origImg = img;
  // create mask canvas same size
  maskCanvas = document.createElement('canvas');
  maskCanvas.width = img.width; maskCanvas.height = img.height;
  maskCtx = maskCanvas.getContext('2d');
  maskCtx.fillStyle='rgba(0,0,0,0)';
  maskCtx.fillRect(0,0,maskCanvas.width,maskCanvas.height);
  drawOverlay();
});

function drawOverlay(){
  const canvas = document.getElementById('canvas');
  if(!origImg) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(origImg,0,0);
  if(maskCanvas){
    // draw mask semi-transparent red
    ctx.globalAlpha = 0.45;
    ctx.drawImage(maskCanvas,0,0);
    ctx.globalAlpha = 1.0;
  }
}

let drawing=false, mode='pencil';
document.getElementById('canvas').addEventListener('pointerdown', (e)=>{
  drawing=true; drawAt(e);
});
document.getElementById('canvas').addEventListener('pointermove', (e)=>{ if(drawing) drawAt(e); });
document.getElementById('canvas').addEventListener('pointerup', ()=>{ drawing=false; });
document.getElementById('pencil').onclick = ()=> mode='pencil';
document.getElementById('eraser').onclick = ()=> mode='eraser';
function drawAt(e){
  if(!maskCtx) return;
  const rect = e.target.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (e.target.width / rect.width);
  const y = (e.clientY - rect.top) * (e.target.height / rect.height);
  const sz = parseInt(document.getElementById('size').value,10);
  if(mode==='pencil'){
    maskCtx.globalCompositeOperation = 'source-over';
    maskCtx.fillStyle = 'rgba(255,0,0,255)';
  } else {
    maskCtx.globalCompositeOperation = 'destination-out';
    maskCtx.fillStyle = 'rgba(0,0,0,255)';
  }
  maskCtx.beginPath(); maskCtx.arc(x,y,sz,0,Math.PI*2); maskCtx.fill();
  drawOverlay();
}

// helper to send form with mask as blob
async function sendWithMask(url){
  const f = document.getElementById('file').files[0];
  if(!f) return alert('Choose image first');
  const fd = new FormData();
  fd.append('file', f);
  fd.append('k', document.querySelector('input[name=k]').value);
  fd.append('pxmm', document.querySelector('input[name=pxmm]').value);
  fd.append('fill', document.querySelector('input[name=fill]').value);
  fd.append('angle', document.querySelector('input[name=angle]').value);
  fd.append('use_onnx', document.querySelector('input[name=use_onnx]').checked ? '1' : '0');
  // export mask PNG
  if(maskCanvas){
    const blob = await new Promise(r=> maskCanvas.toBlob(r,'image/png'));
    fd.append('mask', blob, 'mask.png');
  }
  const res = await fetch(url, {method:'POST', body:fd});
  return res;
}

document.getElementById('previewBtn').addEventListener('click', async ()=> {
  const res = await sendWithMask('/api/preview');
  const data = await res.json();
  document.getElementById('preview').innerHTML = `<img style="max-width:100%" src="${data.image}"/>`;
});

document.getElementById('exportBtn').addEventListener('click', async ()=>{
  const res = await sendWithMask('/api/auto-digitize');
  if(!res.ok) return alert('Failed to generate DST');
  const blob = await res.blob(); const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href=url; a.download='autodigitized-'+Date.now()+'.dst'; a.click();
  URL.revokeObjectURL(url);
});
</script>
</body></html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML

# === API endpoints accept optional mask file (painted) ===
@app.post("/api/preview")
async def api_preview(file: UploadFile = File(...), mask: UploadFile = File(None),
                      k: int = Form(8), pxmm: float = Form(10.0), fill: float = Form(0.5),
                      angle: float = Form(45.0), use_onnx: str = Form("0")):
    img_bytes = await file.read()
    pil = read_image_to_rgb(img_bytes, max_dim=1200)
    bgr = pil_to_cv(pil) if True else np.array(pil)[:,:,::-1]
    seg = None
    # ONNX path
    if use_onnx == "1" and onnx_seg.sess is not None:
        seg = onnx_seg.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if seg is None:
        seg = kmeans_segmentation(bgr, k=k)
    # if mask provided, read it as binary and create mask_override
    mask_override = None
    if mask is not None:
        mbytes = await mask.read()
        mimg = Image.open(BytesIO(mbytes)).convert('L').resize((seg.shape[1], seg.shape[0]))
        mnp = np.array(mimg)
        mask_override = (mnp>10).astype(np.uint8)*255
    pattern = plan_stitches(bgr, seg, px_per_mm=pxmm, fill_spacing_mm=fill, fill_angle=angle, mask_override=mask_override)
    # render preview
    xs, ys = [], []
    for st in pattern.stitches:
        cmd, x, y = st[0], st[1], st[2]
        if cmd == STITCH:
            xs.append(x); ys.append(y)
        else:
            xs.append(None); ys.append(None)
    plt.figure(figsize=(6,6)); plt.plot(xs, ys, linewidth=0.6); plt.gca().invert_yaxis(); plt.axis('equal'); plt.axis('off')
    buf = BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05); plt.close()
    buf.seek(0); b64 = base64.b64encode(buf.read()).decode('utf-8')
    return JSONResponse({"image": f"data:image/png;base64,{b64}"})

@app.post("/api/auto-digitize")
async def api_auto_digitize(file: UploadFile = File(...), mask: UploadFile = File(None),
                            k: int = Form(8), pxmm: float = Form(10.0), fill: float = Form(0.5),
                            angle: float = Form(45.0), use_onnx: str = Form("0")):
    img_bytes = await file.read()
    pil = read_image_to_rgb(img_bytes, max_dim=1200)
    bgr = pil_to_cv(pil)
    seg = None
    if use_onnx == "1" and onnx_seg.sess is not None:
        seg = onnx_seg.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if seg is None:
        seg = kmeans_segmentation(bgr, k=k)
    mask_override = None
    if mask is not None:
        mbytes = await mask.read()
        mimg = Image.open(BytesIO(mbytes)).convert('L').resize((seg.shape[1], seg.shape[0]))
        mnp = np.array(mimg)
        mask_override = (mnp>10).astype(np.uint8)*255
    pattern = plan_stitches(bgr, seg, px_per_mm=pxmm, fill_spacing_mm=fill, fill_angle=angle, mask_override=mask_override)
    out = BytesIO()
    write_dst(pattern, out)
    out.seek(0)
    headers = {"Content-Disposition": f"attachment; filename=autodigitized-{uuid.uuid4().hex[:8]}.dst"}
    return StreamingResponse(out, headers=headers, media_type="application/octet-stream")
