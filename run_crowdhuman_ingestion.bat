python ./siammot/data/ingestion/ingest_crowdhuman.py ../../datasets/CrowdHuman/anno/annotation_train.odgt ../../datasets/CrowdHuman/Images ../../datasets/CrowdHuman_COCO -b fbox -s train

python ./siammot/data/ingestion/ingest_crowdhuman.py ../../datasets/CrowdHuman/anno/annotation_train.odgt ../../datasets/CrowdHuman/Images ../../datasets/CrowdHuman_COCO -b vbox -s train

python ./siammot/data/ingestion/ingest_crowdhuman.py ../../datasets/CrowdHuman/anno/annotation_val.odgt ../../datasets/CrowdHuman/Images ../../datasets/CrowdHuman_COCO -b fbox -s val

python ./siammot/data/ingestion/ingest_crowdhuman.py ../../datasets/CrowdHuman/anno/annotation_val.odgt ../../datasets/CrowdHuman/Images ../../datasets/CrowdHuman_COCO -b vbox -s val