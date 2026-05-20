#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"

create_response="$(curl -sS -X POST "${BASE_URL}/sessions" -H "content-type: application/json" -d '{}')"
session_id="$(printf '%s' "${create_response}" | python -c 'import sys,json; print(json.load(sys.stdin)["session_id"])')"
echo "session_id=${session_id}"

curl -sS -X POST "${BASE_URL}/sessions/${session_id}/predict" -H "content-type: application/json" -d '{}' >/dev/null

frames_response="$(curl -sS "${BASE_URL}/sessions/${session_id}/forecast/latest/frames")"
frame_count="$(printf '%s' "${frames_response}" | python -c 'import sys,json; print(json.load(sys.stdin).get("frame_count",0))')"
default_frame_index="$(printf '%s' "${frames_response}" | python -c 'import sys,json; d=json.load(sys.stdin); idx=d.get("default_frame_index",0); arr=d.get("frame_indices",[]); print(idx if idx in arr else (arr[0] if arr else 0))')"
echo "frame_count=${frame_count}"
echo "selected_frame_index=${default_frame_index}"

geojson_response="$(curl -sS "${BASE_URL}/sessions/${session_id}/forecast/latest/frames/${default_frame_index}/geojson")"
printf '%s' "${geojson_response}" | python -c 'import sys,json; d=json.load(sys.stdin); fs=d.get("features",[]); kinds={}; plume=0; 
for f in fs:
 p=(f or {}).get("properties",{}) or {}
 k=p.get("kind")
 if isinstance(k,str) and k:
  kinds[k]=kinds.get(k,0)+1
  if k=="plume_cell": plume += 1
plume_band=kinds.get("plume_band",0)
print(f"feature_count={len(fs)}")
print("kinds=" + ",".join(sorted(kinds.keys())))
plume_point=kinds.get("plume_point",0)
max_conc=d.get("properties",{}).get("max_concentration")
rendered_points=d.get("properties",{}).get("rendered_point_count")
print(f"plume_point={plume_point}")
print(f"plume_band={plume_band}")
print(f"plume_cell={plume}")
print(f"max_concentration={max_conc}")
print(f"rendered_point_count={rendered_points}")'

raster_response="$(curl -sS "${BASE_URL}/sessions/${session_id}/forecast/latest/frames/${default_frame_index}/raster")"
printf '%s' "${raster_response}" | python -c 'import sys,json; d=json.load(sys.stdin); print(f"raster_width={d.get(\"width\")}"); print(f"raster_height={d.get(\"height\")}"); print(f"raster_min={d.get(\"min_concentration\")}"); print(f"raster_max={d.get(\"max_concentration\")}")'
