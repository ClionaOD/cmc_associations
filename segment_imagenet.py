import pixellib
from pixellib.semantic import semantic_segmentation
from pixellib.instance import instance_segmentation
from pixellib.tune_bg import alter_bg

segment_image = semantic_segmentation()
segment_image.load_pascalvoc_model("./segmentation_models/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5") 
segment_image.segmentAsPascalvoc("./test_imagenet/n07889990/img_4aea68f782e7a304161be93a21def6e5.jpg", 
    output_image_name = "./test_imagenet/beer_segmented.jpg", 
    overlay=True)

segment_image = instance_segmentation()
segment_image.load_model("./segmentation_models/mask_rcnn_coco.h5") 
segment_image.segmentImage("./test_imagenet/n07889990/img_4aea68f782e7a304161be93a21def6e5.jpg", 
    output_image_name = "./test_imagenet/beer_instance_segmented.jpg",
    show_bboxes=True)

change_bg = alter_bg(model_type = "pb")
change_bg.load_pascalvoc_model("./segmentation_models/xception_pascalvoc.pb")
output = change_bg.color_bg("./test_imagenet/n07889990/img_4aea68f782e7a304161be93a21def6e5.jpg", 
    colors = (255,255,255))