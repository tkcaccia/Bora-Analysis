"""Profile selection over fabricated 270-slide metadata, never CAM records."""

from pathlib import Path
import json
import resource
from time import perf_counter

from select_metadata_only import propose


slides = []
for i in range(110):
    slides.append({"slide_id": f"tumor_{i:03d}",
                   "partition": "public_development",
                   "official_listing_verified": True,
                   "reference_kind": "lesion_mask",
                   "patient_id": f"fabricated-tumor-{i:03d}",
                   "center": "A" if i % 2 else "B",
                   "source_sha256": None})
for i in range(160):
    slides.append({"slide_id": f"normal_{i:03d}",
                   "partition": "public_development",
                   "official_listing_verified": True,
                   "reference_kind": "negative_slide_label",
                   "patient_id": f"fabricated-normal-{i:03d}",
                   "center": "B" if i % 2 else "A",
                   "source_sha256": None})
started = perf_counter()
result = propose({"format": "bora-camelyon16-official-metadata-v1",
                  "slides": slides})
profile = {"status": "FABRICATED_METADATA_ONLY", "entries": len(slides),
           "wall_seconds": perf_counter() - started,
           "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           "selected_synthetic_ids": result["test_slide_ids"],
           "patient_relationship_status": result["patient_relationship_status"]}
Path(__file__).with_name("SYNTHETIC_270_METADATA_PROFILE.json").write_text(
    json.dumps(profile, indent=2) + "\n")
print(json.dumps(profile))
