from dataclasses import dataclass

@dataclass
class Base:
    run_name: str
    model: str
    projection: str
    notes: str
