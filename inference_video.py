# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Run a fine-tuned RF-DETR Medium model on a video file."""

from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch

from rfdetr import RFDETRMedium


def main() -> None:
    """Run the fine-tuned detector on a video and save annotations."""
    checkpoint = Path(r"D:\Dre\rf-detr\output\checkpoint_best_total.pth")
    source_video = Path(r"D:\Dre\rf-detr\input.mp4")
    output_video = Path(r"D:\Dre\rf-detr\output\prediction.mp4")

    for path in (checkpoint, source_video):
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

    output_video.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RFDETRMedium.from_checkpoint(str(checkpoint), device=device)
    predictor = model.inference(compile=False)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    def process_frame(frame: np.ndarray, index: int) -> np.ndarray:
        """Predict objects in one frame and draw their boxes and labels."""
        # Video frames use BGR; RF-DETR expects RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = predictor.predict(rgb_frame, threshold=0.5)

        labels = [
            f"{name} {confidence:.2f}"
            for name, confidence in zip(
                detections.data["class_name"],
                detections.confidence,
            )
        ]

        annotated = box_annotator.annotate(
            scene=frame.copy(),
            detections=detections,
        )
        return label_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=labels,
        )

    print(f"Using device: {device}")
    sv.process_video(
        source_path=str(source_video),
        target_path=str(output_video),
        callback=process_frame,
    )
    print(f"Saved video: {output_video}")


if __name__ == "__main__":
    main()
