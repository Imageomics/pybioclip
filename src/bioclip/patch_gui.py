import os
import sys
import io
import json
import base64
import socket
import threading
import webbrowser
import numpy as np
import PIL.Image
from PIL import ImageFilter
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, List, Optional


GRID_SIZE = 14
PATCH_SIZE = 16
IMAGE_SIZE = 224
DATASET_MEAN_RGB = (0.48145466, 0.4578275, 0.40821073)


class MaskStrategy(Enum):
    MEAN_FILL = "mean"
    ZERO_FILL = "zero"
    GAUSSIAN_BLUR = "blur"


class SelectionMode(Enum):
    GRID = "grid"
    FREEFORM = "freeform"


def create_grid_mask_pixels(grid_mask: np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Expand a 14x14 boolean grid mask to a (224, 224) boolean pixel mask.

    Each True cell in the grid becomes a 16x16 block of True pixels.
    """
    patch_size = image_size // GRID_SIZE
    return np.kron(grid_mask, np.ones((patch_size, patch_size), dtype=bool))


def apply_mean_fill(image_array: np.ndarray, pixel_mask: np.ndarray) -> np.ndarray:
    """Replace masked pixels with dataset mean RGB values."""
    result = image_array.copy()
    mean_rgb = np.array([int(c * 255) for c in DATASET_MEAN_RGB], dtype=np.uint8)
    result[pixel_mask] = mean_rgb
    return result


def apply_zero_fill(image_array: np.ndarray, pixel_mask: np.ndarray) -> np.ndarray:
    """Replace masked pixels with 0 (black)."""
    result = image_array.copy()
    result[pixel_mask] = 0
    return result


def apply_gaussian_blur(image: PIL.Image.Image, pixel_mask: np.ndarray, radius: int = 20) -> PIL.Image.Image:
    """Apply heavy Gaussian blur to masked regions, preserving unmasked areas."""
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    result = np.array(image)
    blurred_array = np.array(blurred)
    result[pixel_mask] = blurred_array[pixel_mask]
    return PIL.Image.fromarray(result)


def _kept_bbox(pixel_mask: np.ndarray):
    """Find the bounding box of the kept (False) region in a pixel mask.

    Returns (y0, y1, x0, x1) in pixel coordinates, or None if everything is masked.
    """
    kept = ~pixel_mask
    if not kept.any():
        return None
    rows = np.any(kept, axis=1)
    cols = np.any(kept, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return int(y0), int(y1) + 1, int(x0), int(x1) + 1


def apply_mask_to_image(
    image: PIL.Image.Image,
    mask: np.ndarray,
    strategy: MaskStrategy,
    mode: SelectionMode,
) -> PIL.Image.Image:
    """Apply a mask to an image, cropping to the kept region at full resolution.

    Crops the original image to the bounding box of the unmasked (kept) region
    at its native resolution, applies masking to any interior holes within the
    crop, then resizes to 224x224. This maximizes the resolution of the content
    the model sees.

    Args:
        image: Input PIL Image at any resolution.
        mask: Boolean array — shape (14, 14) for GRID mode, (224, 224) for FREEFORM.
              True = masked (removed), False = kept.
        strategy: Which fill method to use for masked regions.
        mode: Whether mask is grid-level or pixel-level.

    Returns:
        Masked and cropped PIL Image at 224x224.
    """
    if mode == SelectionMode.GRID:
        pixel_mask = create_grid_mask_pixels(mask)
    else:
        pixel_mask = mask

    if not pixel_mask.any():
        # Nothing masked — just resize the full image
        return image.resize((IMAGE_SIZE, IMAGE_SIZE), PIL.Image.LANCZOS)

    bbox = _kept_bbox(pixel_mask)
    if bbox is None:
        # Everything masked — return a fully-filled 224x224 image
        img_224 = image.resize((IMAGE_SIZE, IMAGE_SIZE), PIL.Image.LANCZOS)
        img_array = np.array(img_224)
        fill_mask = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        if strategy == MaskStrategy.GAUSSIAN_BLUR:
            return apply_gaussian_blur(img_224, fill_mask)
        elif strategy == MaskStrategy.MEAN_FILL:
            return PIL.Image.fromarray(apply_mean_fill(img_array, fill_mask))
        else:
            return PIL.Image.fromarray(apply_zero_fill(img_array, fill_mask))

    y0, y1, x0, x1 = bbox
    orig_w, orig_h = image.size

    # Map bounding box from 224x224 mask space to original image coordinates
    scale_x = orig_w / IMAGE_SIZE
    scale_y = orig_h / IMAGE_SIZE
    crop_left = int(x0 * scale_x)
    crop_upper = int(y0 * scale_y)
    crop_right = min(orig_w, int(np.ceil(x1 * scale_x)))
    crop_lower = min(orig_h, int(np.ceil(y1 * scale_y)))

    cropped = image.crop((crop_left, crop_upper, crop_right, crop_lower))

    # Check for interior masked pixels within the bounding box
    interior_mask = pixel_mask[y0:y1, x0:x1]
    if interior_mask.any():
        # Resize the interior mask to match the crop dimensions
        crop_w = crop_right - crop_left
        crop_h = crop_lower - crop_upper
        interior_mask_img = PIL.Image.fromarray(interior_mask.astype(np.uint8) * 255)
        interior_mask_resized = interior_mask_img.resize(
            (crop_w, crop_h), PIL.Image.NEAREST
        )
        crop_mask = np.array(interior_mask_resized) > 127

        if strategy == MaskStrategy.GAUSSIAN_BLUR:
            cropped = apply_gaussian_blur(cropped, crop_mask)
        else:
            crop_array = np.array(cropped)
            if strategy == MaskStrategy.MEAN_FILL:
                crop_array = apply_mean_fill(crop_array, crop_mask)
            elif strategy == MaskStrategy.ZERO_FILL:
                crop_array = apply_zero_fill(crop_array, crop_mask)
            cropped = PIL.Image.fromarray(crop_array)

    return cropped.resize((IMAGE_SIZE, IMAGE_SIZE), PIL.Image.LANCZOS)


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _image_to_base64(image: PIL.Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class _PatchServerState:
    """Shared state between the HTTP handler and the main thread."""

    def __init__(self, image_paths: List[str],
                 predict_callback: Callable[[PIL.Image.Image, str], List[dict]],
                 subset_callback: Optional[Callable[[Optional[str]], dict]] = None):
        self.image_paths = image_paths
        self.predict_callback = predict_callback
        self.subset_callback = subset_callback
        self.results: List[dict] = []
        self.done_event = threading.Event()

    def get_image_data(self, index: int) -> dict:
        path = self.image_paths[index]
        img = PIL.Image.open(path).convert("RGB")
        img_224 = img.resize((IMAGE_SIZE, IMAGE_SIZE), PIL.Image.LANCZOS)
        return {
            "image": _image_to_base64(img_224),
            "filename": os.path.basename(path),
            "index": index,
            "total": len(self.image_paths),
        }

    def handle_predict(self, body: dict) -> List[dict]:
        image_index = body["imageIndex"]
        strategy = MaskStrategy(body["strategy"])
        mode = SelectionMode(body["mode"])

        path = self.image_paths[image_index]
        img = PIL.Image.open(path).convert("RGB")

        if mode == SelectionMode.GRID:
            mask = np.array(body["mask"], dtype=bool)
        else:
            mask_bytes = base64.b64decode(body["mask"])
            mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(
                IMAGE_SIZE, IMAGE_SIZE
            ).astype(bool)

        masked = apply_mask_to_image(img, mask, strategy, mode)
        predictions = self.predict_callback(masked, path)
        self.results.extend(predictions)
        return predictions

    def handle_preview(self, body: dict) -> dict:
        image_index = body["imageIndex"]
        strategy = MaskStrategy(body["strategy"])
        mode = SelectionMode(body["mode"])

        path = self.image_paths[image_index]
        img = PIL.Image.open(path).convert("RGB")

        if mode == SelectionMode.GRID:
            mask = np.array(body["mask"], dtype=bool)
        else:
            mask_bytes = base64.b64decode(body["mask"])
            mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(
                IMAGE_SIZE, IMAGE_SIZE
            ).astype(bool)

        masked = apply_mask_to_image(img, mask, strategy, mode)
        return {"image": _image_to_base64(masked)}

    def handle_subset(self, body: dict) -> dict:
        if self.subset_callback is None:
            return {"error": "Subset filtering is only available in TreeOfLife mode."}
        csv_content = body.get("content")  # None means clear
        return self.subset_callback(csv_content)


def _make_handler(state: _PatchServerState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # suppress request logging

        def do_GET(self):
            if self.path == "/":
                self._send_json_or_html(200, "text/html; charset=utf-8",
                                        _HTML_TEMPLATE.encode("utf-8"))
            elif self.path.startswith("/api/image/"):
                self._handle_get_image()
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/api/predict":
                self._handle_predict()
            elif self.path == "/api/preview":
                self._handle_preview()
            elif self.path == "/api/subset":
                self._handle_subset()
            elif self.path == "/api/done":
                self._handle_done()
            else:
                self.send_error(404)

        def _send_json_or_html(self, code, content_type, body):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self):
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length))

        def _handle_get_image(self):
            try:
                index = int(self.path.split("/")[-1])
                if not 0 <= index < len(state.image_paths):
                    self.send_error(404)
                    return
                data = state.get_image_data(index)
                body = json.dumps(data).encode("utf-8")
                self._send_json_or_html(200, "application/json", body)
            except Exception as e:
                self.send_error(500, str(e))

        def _handle_predict(self):
            try:
                body = self._read_json_body()
                predictions = state.handle_predict(body)
                resp = json.dumps(predictions).encode("utf-8")
                self._send_json_or_html(200, "application/json", resp)
            except Exception as e:
                self.send_error(500, str(e))

        def _handle_preview(self):
            try:
                body = self._read_json_body()
                result = state.handle_preview(body)
                resp = json.dumps(result).encode("utf-8")
                self._send_json_or_html(200, "application/json", resp)
            except Exception as e:
                self.send_error(500, str(e))

        def _handle_subset(self):
            try:
                body = self._read_json_body()
                result = state.handle_subset(body)
                resp = json.dumps(result).encode("utf-8")
                code = 400 if "error" in result else 200
                self._send_json_or_html(code, "application/json", resp)
            except Exception as e:
                resp = json.dumps({"error": str(e)}).encode("utf-8")
                self._send_json_or_html(400, "application/json", resp)

        def _handle_done(self):
            state.done_event.set()
            body = b'{"status":"done"}'
            self._send_json_or_html(200, "application/json", body)

    return Handler


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>bioclip patch</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:#1e1e2e;color:#cdd6f4;padding:20px;min-height:100vh}
header{display:flex;align-items:center;gap:16px;margin-bottom:16px}
header h1{font-size:20px;font-weight:600;color:#89b4fa}
header .info{color:#a6adc8;font-size:14px}
.main{display:flex;gap:24px;margin-bottom:16px;flex-wrap:wrap}
.canvas-area{display:flex;gap:20px;flex-wrap:wrap}
.canvas-wrap{text-align:center}
.canvas-wrap h3{font-size:12px;text-transform:uppercase;letter-spacing:.5px;
  color:#a6adc8;margin-bottom:8px}
canvas{border:1px solid #45475a;border-radius:6px;cursor:crosshair;
  background:#181825;display:block}
#preview-canvas{border-color:#89b4fa;border-width:2px}
.controls{display:flex;flex-direction:column;gap:20px;min-width:180px}
.cgroup h4{font-size:11px;text-transform:uppercase;letter-spacing:.5px;
  color:#a6adc8;margin-bottom:8px}
.cgroup label{display:block;font-size:14px;padding:3px 0;cursor:pointer}
.cgroup label:hover{color:#89b4fa}
input[type=radio]{accent-color:#89b4fa;margin-right:6px}
input[type=range]{width:100%;accent-color:#89b4fa}
.brush-val{font-size:13px;color:#a6adc8}
.toolbar{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
button{padding:8px 18px;border:1px solid #45475a;border-radius:6px;
  background:#313244;color:#cdd6f4;cursor:pointer;font-size:14px;
  transition:background .15s}
button:hover{background:#45475a}
button:disabled{opacity:.4;cursor:not-allowed}
button.primary{background:#89b4fa;color:#1e1e2e;border-color:#89b4fa;font-weight:600}
button.primary:hover{background:#74c7ec}
button.done{background:#a6e3a1;color:#1e1e2e;border-color:#a6e3a1;
  font-weight:600;margin-left:auto}
button.done:hover{background:#94e2d5}
.results{background:#313244;border-radius:6px;padding:16px}
.results h3{font-size:12px;text-transform:uppercase;letter-spacing:.5px;
  color:#a6adc8;margin-bottom:10px}
.results table{width:100%;border-collapse:collapse;font-size:13px;
  font-family:'SF Mono',Consolas,monospace}
.results td,.results th{padding:4px 10px;text-align:left}
.results th{color:#a6adc8;font-weight:500;border-bottom:1px solid #45475a}
.results tr:hover td{background:#45475a33}
.score-bar{display:inline-block;height:8px;border-radius:4px;
  background:#89b4fa;vertical-align:middle;margin-right:6px}
.empty-msg{color:#6c7086;font-style:italic;font-size:13px}
.spinner{display:inline-block;width:16px;height:16px;border:2px solid #45475a;
  border-top-color:#89b4fa;border-radius:50%;animation:spin .6s linear infinite;
  vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
#brush-group{opacity:.4;transition:opacity .2s}
#brush-group.active{opacity:1}
.file-btn{display:inline-block;padding:5px 12px;background:#313244;border:1px solid #45475a;
  border-radius:4px;cursor:pointer;font-size:13px;transition:background .15s}
.file-btn:hover{background:#45475a}
.subset-status{display:block;font-size:12px;color:#a6e3a1;margin-top:4px;
  word-break:break-all}
.subset-status.error{color:#f38ba8}
.small-btn{padding:3px 10px;font-size:12px;margin-top:4px}
</style>
</head>
<body>
<header>
  <h1>bioclip patch</h1>
  <span class="info" id="img-info">Loading...</span>
</header>

<div class="main">
  <div class="canvas-area">
    <div class="canvas-wrap">
      <h3>Select patches to mask</h3>
      <canvas id="main-canvas" width="448" height="448"
              style="width:448px;height:448px"></canvas>
    </div>
    <div class="canvas-wrap">
      <h3>Model Input (224&times;224)</h3>
      <canvas id="preview-canvas" width="448" height="448"
              style="width:448px;height:448px"></canvas>
    </div>
  </div>

  <div class="controls">
    <div class="cgroup">
      <h4>Mode</h4>
      <label><input type="radio" name="mode" value="grid" checked>Grid (14&times;14)</label>
      <label><input type="radio" name="mode" value="freeform">Freeform</label>
    </div>
    <div class="cgroup">
      <h4>Mask Fill</h4>
      <label><input type="radio" name="strategy" value="mean" checked>Mean fill</label>
      <label><input type="radio" name="strategy" value="zero">Zero fill (black)</label>
      <label><input type="radio" name="strategy" value="blur">Gaussian blur</label>
    </div>
    <div class="cgroup" id="brush-group">
      <h4>Brush Size</h4>
      <input type="range" id="brush-slider" min="1" max="30" value="10">
      <span class="brush-val" id="brush-val">10</span>
    </div>
    <div class="cgroup">
      <h4>Subset Filter</h4>
      <label class="file-btn" id="subset-label">
        <input type="file" id="subset-file" accept=".csv,.txt" style="display:none">
        Choose file...
      </label>
      <span class="subset-status" id="subset-status"></span>
      <button id="btn-clear-subset" class="small-btn" style="display:none">Clear</button>
    </div>
  </div>
</div>

<div class="toolbar">
  <button id="btn-prev">&laquo; Prev</button>
  <button id="btn-clear">Clear</button>
  <button id="btn-invert">Invert</button>
  <button id="btn-predict" class="primary">Predict</button>
  <button id="btn-next">Next &raquo;</button>
  <button id="btn-done" class="done">Done</button>
</div>

<div class="results">
  <h3>Predictions</h3>
  <div id="results-content"><span class="empty-msg">Select patches and click Predict</span></div>
</div>

<script>
(function(){
"use strict";
const S=224, G=14, D=448, PD=D/G, SCALE=D/S;
const MEAN=[123,117,104];

const mainC=document.getElementById("main-canvas");
const prevC=document.getElementById("preview-canvas");
const mCtx=mainC.getContext("2d");
const pCtx=prevC.getContext("2d");

let gridMask=make2d(G,G,false);
let freeMask=new Uint8Array(S*S);
let mode="grid", strategy="mean", brush=10;
let idx=0, total=1, isPainting=false;
let baseImg=new Image();

function make2d(r,c,v){return Array.from({length:r},()=>Array(c).fill(v))}

function getCoords(e){
  const r=mainC.getBoundingClientRect();
  const x=Math.floor((e.clientX-r.left)*S/r.width);
  const y=Math.floor((e.clientY-r.top)*S/r.height);
  return{x:Math.max(0,Math.min(S-1,x)),y:Math.max(0,Math.min(S-1,y))};
}

function loadImage(i){
  fetch("/api/image/"+i).then(r=>r.json()).then(d=>{
    total=d.total; idx=d.index;
    gridMask=make2d(G,G,false);
    freeMask=new Uint8Array(S*S);
    baseImg.onload=()=>{redraw();updatePreview()};
    baseImg.src="data:image/png;base64,"+d.image;
    document.getElementById("img-info").textContent=
      d.filename+" ("+(idx+1)+"/"+total+")";
    document.title="bioclip patch - "+d.filename;
    document.getElementById("btn-prev").disabled=(idx===0);
    document.getElementById("btn-next").disabled=(idx>=total-1);
  });
}

function redraw(){
  mCtx.imageSmoothingEnabled=true;
  mCtx.drawImage(baseImg,0,0,D,D);
  if(mode==="grid") drawGrid(); else drawFreeform();
}

function drawGrid(){
  mCtx.strokeStyle="rgba(255,255,255,0.15)";
  mCtx.lineWidth=1;
  for(let i=1;i<G;i++){
    const p=i*PD;
    mCtx.beginPath();mCtx.moveTo(p,0);mCtx.lineTo(p,D);mCtx.stroke();
    mCtx.beginPath();mCtx.moveTo(0,p);mCtx.lineTo(D,p);mCtx.stroke();
  }
  mCtx.fillStyle="rgba(239,68,68,0.35)";
  mCtx.strokeStyle="rgba(239,68,68,0.7)";
  mCtx.lineWidth=1.5;
  for(let r=0;r<G;r++) for(let c=0;c<G;c++) if(gridMask[r][c]){
    mCtx.fillRect(c*PD,r*PD,PD,PD);
    mCtx.strokeRect(c*PD+.5,r*PD+.5,PD-1,PD-1);
  }
}

function drawFreeform(){
  const tmp=document.createElement("canvas");tmp.width=S;tmp.height=S;
  const tc=tmp.getContext("2d");
  const id=tc.createImageData(S,S);
  for(let i=0;i<S*S;i++) if(freeMask[i]){
    const j=i*4; id.data[j]=239;id.data[j+1]=68;id.data[j+2]=68;id.data[j+3]=89;
  }
  tc.putImageData(id,0,0);
  mCtx.drawImage(tmp,0,0,D,D);
}

let _previewTimer=null;
let _previewSeq=0;
function updatePreview(){
  clearTimeout(_previewTimer);
  _previewTimer=setTimeout(_fetchPreview, 150);
}
function _fetchPreview(){
  let maskData;
  if(mode==="grid") maskData=gridMask;
  else{let bin="";for(let i=0;i<freeMask.length;i++) bin+=String.fromCharCode(freeMask[i]);
    maskData=btoa(bin);}
  const seq=++_previewSeq;
  fetch("/api/preview",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({mask:maskData,strategy:strategy,mode:mode,imageIndex:idx})
  }).then(r=>r.json()).then(d=>{
    if(seq!==_previewSeq) return;
    const img=new Image();
    img.onload=()=>{pCtx.imageSmoothingEnabled=true;pCtx.drawImage(img,0,0,D,D)};
    img.src="data:image/png;base64,"+d.image;
  });
}

function paintCircle(cx,cy){
  const b=brush;
  for(let y=Math.max(0,cy-b);y<=Math.min(S-1,cy+b);y++)
    for(let x=Math.max(0,cx-b);x<=Math.min(S-1,cx+b);x++)
      if((x-cx)*(x-cx)+(y-cy)*(y-cy)<=b*b) freeMask[y*S+x]=1;
}

// --- Events ---
mainC.addEventListener("mousedown",e=>{
  const{x,y}=getCoords(e);
  if(mode==="grid"){
    const c=Math.min(G-1,Math.floor(x/16));
    const r=Math.min(G-1,Math.floor(y/16));
    gridMask[r][c]=!gridMask[r][c];
    redraw(); updatePreview();
  } else {
    isPainting=true;
    paintCircle(x,y); redraw(); updatePreview();
  }
});
mainC.addEventListener("mousemove",e=>{
  if(!isPainting||mode!=="freeform") return;
  const{x,y}=getCoords(e);
  paintCircle(x,y); redraw(); updatePreview();
});
window.addEventListener("mouseup",()=>{isPainting=false});

document.querySelectorAll('input[name=mode]').forEach(r=>
  r.addEventListener("change",e=>{
    mode=e.target.value;
    document.getElementById("brush-group").classList.toggle("active",mode==="freeform");
    redraw(); updatePreview();
  }));
document.querySelectorAll('input[name=strategy]').forEach(r=>
  r.addEventListener("change",e=>{strategy=e.target.value; updatePreview()}));
document.getElementById("brush-slider").addEventListener("input",e=>{
  brush=parseInt(e.target.value);
  document.getElementById("brush-val").textContent=brush;
});

document.getElementById("btn-clear").addEventListener("click",()=>{
  gridMask=make2d(G,G,false); freeMask=new Uint8Array(S*S);
  redraw(); updatePreview();
});
document.getElementById("btn-invert").addEventListener("click",()=>{
  if(mode==="grid") for(let r=0;r<G;r++) for(let c=0;c<G;c++) gridMask[r][c]=!gridMask[r][c];
  else for(let i=0;i<S*S;i++) freeMask[i]=freeMask[i]?0:1;
  redraw(); updatePreview();
});
document.getElementById("btn-prev").addEventListener("click",()=>{if(idx>0) loadImage(idx-1)});
document.getElementById("btn-next").addEventListener("click",()=>{if(idx<total-1) loadImage(idx+1)});

document.getElementById("btn-predict").addEventListener("click",()=>{
  const btn=document.getElementById("btn-predict");
  btn.disabled=true;
  btn.innerHTML='<span class="spinner"></span>Predicting...';
  let maskData;
  if(mode==="grid") maskData=gridMask;
  else {
    let bin=""; for(let i=0;i<freeMask.length;i++) bin+=String.fromCharCode(freeMask[i]);
    maskData=btoa(bin);
  }
  fetch("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({mask:maskData,strategy:strategy,mode:mode,imageIndex:idx})
  }).then(r=>r.json()).then(preds=>{
    btn.disabled=false; btn.textContent="Predict";
    showResults(preds);
  }).catch(err=>{
    btn.disabled=false; btn.textContent="Predict";
    document.getElementById("results-content").innerHTML=
      '<span class="empty-msg">Error: '+err.message+'</span>';
  });
});

document.getElementById("btn-done").addEventListener("click",()=>{
  fetch("/api/done",{method:"POST"}).then(()=>{
    document.body.innerHTML='<div style="display:flex;align-items:center;'+
      'justify-content:center;height:80vh;color:#a6adc8;font-size:18px">'+
      'Done. You may close this tab.</div>';
  });
});

document.getElementById("subset-file").addEventListener("change",e=>{
  const file=e.target.files[0];
  if(!file) return;
  const reader=new FileReader();
  reader.onload=ev=>{
    const st=document.getElementById("subset-status");
    const btn=document.getElementById("btn-clear-subset");
    st.textContent="Loading..."; st.className="subset-status";
    fetch("/api/subset",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({content:ev.target.result,filename:file.name})
    }).then(r=>r.json()).then(d=>{
      if(d.error){st.textContent=d.error;st.className="subset-status error";btn.style.display="none"}
      else{st.textContent=file.name+" ("+d.count+" taxa)";st.className="subset-status";
        btn.style.display="inline-block";
        document.getElementById("subset-label").textContent="Change file..."}
    }).catch(err=>{st.textContent="Error: "+err.message;st.className="subset-status error"});
  };
  reader.readAsText(file);
});
document.getElementById("btn-clear-subset").addEventListener("click",()=>{
  const st=document.getElementById("subset-status");
  fetch("/api/subset",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({content:null})
  }).then(r=>r.json()).then(d=>{
    st.textContent="";st.className="subset-status";
    document.getElementById("btn-clear-subset").style.display="none";
    document.getElementById("subset-label").textContent="Choose file...";
    document.getElementById("subset-file").value="";
  });
});

function showResults(preds){
  const el=document.getElementById("results-content");
  if(!preds.length){el.innerHTML='<span class="empty-msg">No predictions</span>';return}
  const maxScore=Math.max(...preds.map(p=>p.score||0));
  let html='<table><tr><th>Classification</th><th>Score</th><th></th></tr>';
  preds.forEach(p=>{
    const name=p.classification||p.species||"";
    const score=(p.score||0);
    const pct=maxScore>0?score/maxScore*100:0;
    html+='<tr><td>'+name+'</td><td>'+score.toFixed(4)+'</td>'+
      '<td><span class="score-bar" style="width:'+pct+'px"></span></td></tr>';
  });
  html+='</table>';
  el.innerHTML=html;
}

loadImage(0);
})();
</script>
</body>
</html>"""


def run_patch_gui(image_paths: List[str], predict_kwargs: dict) -> List[dict]:
    """Entry point called from __main__.py. Launches the patch selection GUI.

    Starts a local HTTP server, opens the browser, and waits for the user
    to finish selecting patches and running predictions.
    """
    from bioclip.predict import TreeOfLifeClassifier, CustomLabelsClassifier, CustomLabelsBinningClassifier
    from bioclip.commands import parse_bins_csv

    cls_str = predict_kwargs.get("cls_str")
    rank = predict_kwargs.get("rank")
    bins_path = predict_kwargs.get("bins_path")
    k = predict_kwargs.get("k", 5)
    device = predict_kwargs.get("device", "cpu")
    model_str = predict_kwargs.get("model_str")
    pretrained_str = predict_kwargs.get("pretrained_str")
    subset = predict_kwargs.get("subset")

    model_kwargs = dict(device=device)
    if model_str:
        model_kwargs["model_str"] = model_str
    if pretrained_str:
        model_kwargs["pretrained_str"] = pretrained_str

    # Build classifier once
    if cls_str:
        classifier = CustomLabelsClassifier(cls_ary=cls_str.split(","), **model_kwargs)
    elif bins_path:
        cls_to_bin = parse_bins_csv(bins_path)
        classifier = CustomLabelsBinningClassifier(cls_to_bin=cls_to_bin, **model_kwargs)
    else:
        classifier = TreeOfLifeClassifier(**model_kwargs)
        if subset:
            taxa_filter = classifier.create_taxa_filter_from_csv(subset)
            classifier.apply_filter(taxa_filter)

    def predict_callback(masked_image: PIL.Image.Image, original_path: str) -> List[dict]:
        if cls_str or bins_path:
            predictions = classifier.predict(images=[masked_image], k=k)
        else:
            predictions = classifier.predict(images=[masked_image], rank=rank, k=k)
        for pred in predictions:
            pred["file_name"] = original_path
        return predictions

    subset_callback = None
    if isinstance(classifier, TreeOfLifeClassifier):
        import tempfile
        import pandas as pd

        def subset_callback(csv_content: Optional[str]) -> dict:
            if csv_content is None:
                classifier._subset_txt_embeddings = None
                classifier._subset_txt_names = None
                return {"status": "cleared"}
            # Write content to temp file for create_taxa_filter_from_csv
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                             delete=False, encoding="utf-8") as f:
                f.write(csv_content)
                tmp_path = f.name
            try:
                taxa_filter = classifier.create_taxa_filter_from_csv(tmp_path)
                classifier.apply_filter(taxa_filter)
                count = sum(1 for x in taxa_filter if x)
                return {"status": "applied", "count": count}
            finally:
                os.unlink(tmp_path)

    state = _PatchServerState(image_paths, predict_callback, subset_callback)
    port = _find_free_port()
    handler_cls = _make_handler(state)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://127.0.0.1:{port}"
    print(f"Opening patch selector at {url}")
    webbrowser.open(url)

    try:
        while not state.done_event.wait(timeout=0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()

    return state.results
