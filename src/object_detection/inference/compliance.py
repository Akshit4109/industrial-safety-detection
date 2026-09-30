"""Conservative geometric PPE-to-person association for inference results."""
from typing import Any, Dict, List


def _center(box: Dict[str, float]) -> tuple[float, float]:
    return ((box["xmin"] + box["xmax"]) / 2, (box["ymin"] + box["ymax"]) / 2)


def _belongs_to_person(ppe: Dict[str, float], person: Dict[str, float], upper_body: bool = False) -> bool:
    x, y = _center(ppe)
    if not (person["xmin"] <= x <= person["xmax"] and person["ymin"] <= y <= person["ymax"]):
        return False
    return not upper_body or y <= person["ymin"] + (person["ymax"] - person["ymin"]) * 0.45


def assess_compliance(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Associate detections geometrically; it is not a safety-enforcement decision."""
    people = [d for d in detections if d["class_name"] == "person"]
    helmets = [d for d in detections if d["class_name"] == "helmet"]
    vests = [d for d in detections if d["class_name"] == "vest"]
    workers = []
    for index, person in enumerate(people, 1):
        helmet = any(_belongs_to_person(item["box"], person["box"], upper_body=True) for item in helmets)
        vest = any(_belongs_to_person(item["box"], person["box"]) for item in vests) if vests else None
        workers.append({"person_id": index, "person_box": person["box"], "helmet": helmet, "vest": vest,
                        "status": "COMPLIANT" if helmet and vest is not False else "VIOLATION"})
    return {"people_detected": len(people), "workers": workers,
            "limitations": "Heuristic box-center association. This dataset has no vest class, so vest can be null."}
