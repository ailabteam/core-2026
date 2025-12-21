"""
Data models cho seal và signature detection
"""
from typing import List, Tuple


class BoundingBox:
    """Bounding box cho đối tượng được nhận diện"""
    def __init__(self, x1: int, y1: int, x2: int, y2: int, label: str, confidence: float = 1.0, description: str = ""):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.label = label
        self.confidence = confidence
        self.description = description
        
    def __repr__(self):
        return f"BoundingBox({self.label}: [{self.x1}, {self.y1}, {self.x2}, {self.y2}], conf={self.confidence:.2f})"
    
    def to_dict(self):
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "label": self.label,
            "confidence": self.confidence,
            "description": self.description
        }
    
    def get_coords(self) -> Tuple[int, int, int, int]:
        """Trả về (x1, y1, x2, y2)"""
        return (self.x1, self.y1, self.x2, self.y2)
    
    def get_size(self) -> Tuple[int, int]:
        """Trả về (width, height)"""
        return (self.x2 - self.x1, self.y2 - self.y1)
    
    def get_area(self) -> int:
        """Trả về diện tích của bounding box"""
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class DetectionResult:
    """Kết quả nhận diện con dấu và chữ ký"""
    def __init__(self):
        self.seals: List[BoundingBox] = []
        self.signatures: List[BoundingBox] = []
        self.page: int = 0
        self.width: int = 0
        self.height: int = 0
        
    def add_seal(self, bbox: BoundingBox):
        self.seals.append(bbox)
    
    def add_signature(self, bbox: BoundingBox):
        self.signatures.append(bbox)
    
    def get_all(self) -> List[BoundingBox]:
        """Trả về tất cả các bounding boxes"""
        return self.seals + self.signatures
    
    def to_dict(self):
        return {
            "page": self.page,
            "width": self.width,
            "height": self.height,
            "seals": [s.to_dict() for s in self.seals],
            "signatures": [s.to_dict() for s in self.signatures],
            "total_seals": len(self.seals),
            "total_signatures": len(self.signatures)
        }
    
    def to_json_format(self):
        """
        Trả về định dạng JSON theo schema mới (với bbox là array)
        """
        return {
            "page": self.page,
            "width": self.width,
            "height": self.height,
            "seals": [
                {
                    "type": "seal",
                    "bbox": [s.x1, s.y1, s.x2, s.y2],
                    "confidence": s.confidence,
                    "description": s.description
                }
                for s in self.seals
            ],
            "signatures": [
                {
                    "type": "signature",
                    "bbox": [s.x1, s.y1, s.x2, s.y2],
                    "confidence": s.confidence,
                    "description": s.description
                }
                for s in self.signatures
            ]
        }
    
    def __repr__(self):
        return f"DetectionResult(seals={len(self.seals)}, signatures={len(self.signatures)})"
