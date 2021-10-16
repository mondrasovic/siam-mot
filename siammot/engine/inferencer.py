import logging
import os
import time

import numpy as np
import torch

from PIL import Image
from gluoncv.torch.data.gluoncv_motion_dataset.dataset import DataSample
from tqdm import tqdm

from ..data.adapters.augmentation.build_augmentation import \
    build_siam_augmentation
from ..data.build_inference_data_loader import build_video_loader
from ..eval.eval_clears_mot import eval_clears_mot
from ..utils.boxlists_to_entities import (
    boxlists_to_entities,
    convert_given_detections_to_boxlist,
)
from siammot.modelling.reid.reid_man import build_or_get_existing_reid_manager
from siammot.modelling.track_head.track_solver_debug import (
    build_or_get_existing_track_solver_debugger
)


def build_inverse_tensor_to_pil_transform(cfg):
    pixel_mean = cfg.INPUT.PIXEL_MEAN
    pixel_std = cfg.INPUT.PIXEL_STD

    def _img_tensor_to_pil(img: torch.Tensor) -> Image.Image:
        img = torch.squeeze(img).cpu().detach().numpy()  # [3, H, W]
        img = img.transpose(1, 2, 0)  # [H, W, 3]
        img = (((img * pixel_std) + pixel_mean) * 255).round()
        img = img.astype(np.uint8)
        img = Image.fromarray(img)

        return img
    
    return _img_tensor_to_pil


def do_inference(
    cfg,
    model,
    sample: DataSample,
    transforms=None,
    given_detection: DataSample = None
) -> DataSample:
    """
    Do inference on a specific video (sample)
    :param cfg: configuration file of the model
    :param model: a pytorch model
    :param sample: a testing video
    :param transforms: image-wise transform that prepares
           video frames for processing
    :param given_detection: the cached detections from other model,
           it means that the detection branch is disabled in the
           model forward pass
    :return: the detection results in the format of DataSample
    """
    logger = logging.getLogger(__name__)
    model.eval()
    gpu_device = torch.device('cuda')
    
    reid_manager = build_or_get_existing_reid_manager(cfg)
    reid_manager.reset()
    inverse_img_transform = build_inverse_tensor_to_pil_transform(cfg)

    solver_debugger = build_or_get_existing_track_solver_debugger()
    solver_debugger.reset()
    solver_debugger.sample_width = sample.width
    solver_debugger.sample_height = sample.height

    video_loader = build_video_loader(cfg, sample, transforms)
    
    sample_result = DataSample(
        sample.id, raw_info=None, metadata=sample.metadata
    )
    network_time = 0
    for (video_clip, frame_id, timestamps) in tqdm(video_loader):
        frame_id = frame_id.item()
        timestamps = torch.squeeze(timestamps, dim=0).tolist()
        video_clip = torch.squeeze(video_clip, dim=0)

        frame_orig = inverse_img_transform(video_clip)
        reid_manager.add_next_frame(frame_orig)

        frame_detection = None
        # used the public provided detection (e.g. MOT17, HiEve)
        # the public detection needs to be ingested to DataSample
        # the ingested detection has been provided, find the details in
        # readme/DATA.md
        if given_detection:
            frame_detection = given_detection.get_entities_for_frame_num(
                frame_id
            )
            frame_detection = convert_given_detections_to_boxlist(
                frame_detection,
                sample.width,
                sample.height
            )
            frame_height, frame_width = video_clip.shape[-2:]
            frame_detection = frame_detection.resize(
                (frame_width, frame_height)
            )
            frame_detection = [frame_detection.to(gpu_device)]
        
        with torch.no_grad():
            video_clip = video_clip.to(gpu_device)
            torch.cuda.synchronize()
            network_start_time = time.time()
            output_boxlists = model(video_clip, given_detection=frame_detection)
            torch.cuda.synchronize()
            network_time += time.time() - network_start_time
        
        # Resize to original image size and to xywh mode
        output_boxlists = [
            o.resize([sample.width, sample.height]).convert('xywh')
            for o in output_boxlists]
        output_boxlists = [o.to(torch.device("cpu")) for o in output_boxlists]
        output_entities = boxlists_to_entities(
            output_boxlists, frame_id, timestamps
        )
        for entity in output_entities:
            sample_result.add_entity(entity)
        
        reid_manager.increment_frame_idx()
    
    logger.info(
        'Sample_id {} / Speed {} fps'.format(
            sample.id, len(sample) / (network_time)
        )
    )

    # TODO Implement better path handling.
    solver_debugger.save_to_file(f"track_solver_debug_{sample.id}.json")
    
    return sample_result


class DatasetInference(object):
    def __init__(
        self,
        cfg,
        model,
        dataset,
        output_dir,
        data_filter_fn=None,
        public_detection=None,
        distributed=False,
        motsummary_csv_file_path=None
    ): 
        self._cfg = cfg
        
        self._transform = build_siam_augmentation(cfg, is_train=False)
        self._model = model
        self._dataset = dataset
        self._output_dir = output_dir
        self._distributed = distributed
        self._data_filter_fn = data_filter_fn
        self._pub_detection = public_detection
        self._track_conf = 0.7
        self._track_len = 5
        self._logger = logging.getLogger(__name__)
        self._motsummary_csv_file_path = motsummary_csv_file_path

        self.results = dict()
    
    def _eval_det_ap(self):
        from ..eval.eval_det_ap import eval_det_ap
        iou_threshold = np.arange(0.5, 0.95, 0.05).tolist()
        # TODO Supply class table.
        ap_matrix = eval_det_ap(
            self._dataset, self.results,
            data_filter_fn=self._data_filter_fn,
            iou_threshold=iou_threshold
        )
        ap = np.mean(ap_matrix, axis=0)
        
        ap_str_summary = "\n"
        ap_str_summary += 'Detection AP @[ IoU=0.50:0.95 ] = {:.2f}\n'.format(
            np.mean(ap) * 100
        )
        ap_str_summary += 'Detection AP @[ IoU=0.50 ] = {:.2f}\n'.format(
            ap[0] * 100
        )
        ap_str_summary += 'Detection AP @[ IoU=0.75 ] = {:.2f}\n'.format(
            ap[5] * 100
        )
        
        return ap, ap_str_summary
    
    def _eval_clear_mot(self):
        motmetric, motsummary, motstrsummary = eval_clears_mot(
            self._dataset, self.results,
            data_filter_fn=self._data_filter_fn
        )
        return motmetric, motsummary, motstrsummary
    
    def _inference_on_video(self, sample):
        cache_path = os.path.join(self._output_dir, '{}.json'.format(sample.id))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        if os.path.exists(cache_path):
            sample_result = DataSample.load(cache_path)
        else:
            given_detection = None
            if self._pub_detection:
                given_detection = self._pub_detection[sample.id]
            sample_result = do_inference(
                self._cfg, self._model, sample,
                transforms=self._transform,
                given_detection=given_detection
            )
            sample_result.dump(cache_path)
        return sample_result
    
    def _postprocess_tracks(self, tracks: DataSample):
        """
        post_process the tracks to filter out short and non-confident tracks
        :param tracks: un-filtered tracks
        :return: filtered tracks that would be used for evaluation
        """
        track_ids = set()
        for _entity in tracks.entities:
            if (_entity.id not in track_ids) and (_entity.id >= 0):
                track_ids.add(_entity.id)
        
        filter_tracks = tracks.get_copy_without_entities()
        for _id in track_ids:
            _id_entities = tracks.get_entities_with_id(_id)
            _track_conf = np.mean([_e.confidence for _e in _id_entities])
            if (
                (len(_id_entities) >= self._track_len) and
                (_track_conf >= self._track_conf)
            ):
                for _entity in _id_entities:
                    filter_tracks.add_entity(_entity)
        return filter_tracks
    
    def __call__(self):
        # TODO: Enable the inference in an efficient distributed framework.
        for _, sample in tqdm(self._dataset):
            # clean up the memory
            self._model.reset_siammot_status()
            
            sample_result = self._inference_on_video(sample)
            
            sample_result = self._postprocess_tracks(sample_result)
            self.results.update({sample.id: sample_result})
        
        self._logger.info(
            "\n---------------- Start evaluating ----------------\n"
        )
        _, motsummary, motstrsummary = self._eval_clear_mot()
        self._logger.info(motstrsummary)
        
        self._logger.info(
            "\n---------------- Finish evaluating ----------------\n"
        )

        if self._motsummary_csv_file_path is not None:
            motsummary.to_csv(self._motsummary_csv_file_path)
