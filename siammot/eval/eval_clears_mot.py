import motmetrics as mm

from tqdm import tqdm


def eval_clears_mot(
    samples,
    predicted_samples,
    data_filter_fn=None,
    iou_thresh=0.5
):
    """
    :param samples: a list of (sample_id, sample:DataSample)
    :param predicted_samples: a dict with (sample_id:
    predicted_tracks:DataSample)
    :param data_filter_fn: a callable function to filter entities
    :param iou_thresh: The IOU (between a predicted bounding box and gt )
    threshold that determines a predicted bounding box is a true positive
    """
    
    assert 0 < iou_thresh <= 1
    
    all_accumulators = []
    sample_ids = []
    
    metrics_host = mm.metrics.create()
    
    for (sample_id, sample) in tqdm(samples):
        
        predicted_tracks = predicted_samples[sample_id]
        num_frames = len(sample)
        
        def get_id_and_bbox(entities):
            ids = [entity.id for entity in entities]
            bboxes = [entity.bbox for entity in entities]
            return ids, bboxes
        
        accumulator = mm.MOTAccumulator(auto_id=True)
        for i in range(num_frames):
            valid_gt = sample.get_entities_for_frame_num(i)
            ignore_gt = []
            
            # If data filter function is available
            if data_filter_fn is not None:
                valid_gt, ignore_gt = data_filter_fn(
                    valid_gt,
                    meta_data=sample.metadata
                )
            gt_ids, gt_bboxes = get_id_and_bbox(valid_gt)
            out_ids = []
            out_bboxes = []
            
            # if there is no annotation for a particular frame, we don't
            # evaluate on it
            # this happens for low-fps annotation such as in CRP
            # if len(gt_bboxes) > 0:
            predicted_entities = predicted_tracks.get_entities_for_frame_num(i)
            
            # If data filter function is available
            if data_filter_fn is not None:
                valid_pred, _ = data_filter_fn(
                    predicted_entities, ignore_gt
                )
            else:
                valid_pred = predicted_entities
            
            out_ids, out_bboxes = get_id_and_bbox(valid_pred)
            
            bbox_distances = mm.distances.iou_matrix(
                gt_bboxes, out_bboxes, max_iou=1 - iou_thresh
            )
            accumulator.update(gt_ids, out_ids, bbox_distances)
        
        all_accumulators.append(accumulator)
        sample_ids.append(sample_id)
    
    # Make sure to update to the latest version of motmetrics via pip or idf1
    # calculation might be very slow
    metrics = [
        'num_frames',  # Total number of frames.
        'num_matches',  # Total number of matches.
        'num_switches',  # Total number of track switches.
        'num_false_positives',  # Total number of false positives (false alarms).
        'num_misses',  # Total number of misses (false negatives).
        'num_objects',  # Total number of unique object appearances over all frames (ground-truth).
        'num_predictions',  # Total number of unique prediction appearances over all frames.
        'num_fragmentations',  # Total number of switches from tracked to not tracked.
        'mostly_tracked',  # Number of objects tracked for at least 80 percent of lifespan.
        'partially_tracked',  # Number of objects tracked between 20 and 80 percent of lifespan.
        'precision',  # Number of detected objects over sum of detected and false positives.
        'recall',  # Number of detections over number of objects.
        'idf1',  # ID measures: global min-cost F1 score.
        'mota',  # Multiple object tracker accuracy.
        'motp',  # Multiple object tracker precision.
    ]
    
    summary = None
    strsummary = ""
    if len(all_accumulators):
        summary = metrics_host.compute_many(
            all_accumulators,
            metrics=metrics,
            names=sample_ids,
            generate_overall=True
        )
        
        strsummary = mm.io.render_summary(
            summary,
            formatters=metrics_host.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
    
    return all_accumulators, summary, "\n\n" + strsummary + "\n\n"
