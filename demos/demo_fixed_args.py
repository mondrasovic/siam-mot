import logging
import os

from demos.demo_inference import DemoInference
from demos.utils.vis_generator import VisGenerator
from demos.utils.vis_writer import VisWriter
from demos.video_iterator import build_video_iterator

from dataclasses import dataclass

if __name__ == '__main__':
    @dataclass(frozen=True)
    class Args:
        vis_resolution: int = 1080
        track_class: str = 'person_vehicle'
        dump_video: bool = False
        output_path: str = "E:/siammot_sandbox/tracking_vis"
        demo_video: str = "E:/interreg_sample.mp4"
    
    args = Args()
    
    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%('
               'lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO
    )
    
    # Build visualization generator and writer
    vis_generator = VisGenerator(vis_height=args.vis_resolution)
    vis_writer = VisWriter(
        dump_video=args.dump_video,
        out_path=args.output_path,
        file_name=os.path.basename(args.demo_video)
    )
    
    # Build demo inference
    tracker = DemoInference(
        track_class=args.track_class,
        vis_generator=vis_generator,
        vis_writer=vis_writer
    )
    
    # Build video iterator for inference
    video_reader = build_video_iterator(args.demo_video)
    
    results = list(tracker.process_frame_sequence(video_reader()))
    
    if args.dump_video:
        vis_writer.close_video_writer()
