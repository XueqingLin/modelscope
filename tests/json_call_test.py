import os

from modelscope.hub.api import HubApi
from modelscope.hub.file_download import model_file_download
from modelscope.hub.utils.utils import get_cache_dir
from modelscope.pipelines import pipeline
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile
from modelscope.utils.input_output import (
    call_pipeline_with_json, get_pipeline_information_by_pipeline,
    get_task_input_examples, pipeline_output_to_service_base64_output)


class ModelJsonTest:

    def __init__(self):
        self.api = HubApi()

    def test_single(self, model_id: str, model_revision=None):
        # get model_revision & task info
        cache_root = get_cache_dir()
        print(f"cache_root is {cache_root}")
        configuration_file = os.path.join(cache_root, model_id,
                                          ModelFile.CONFIGURATION)
        print(f"configuration_file is {configuration_file}")
        if not model_revision:
            model_revision = self.api.list_model_revisions(
                model_id=model_id)[0]
        if not os.path.exists(configuration_file):

            configuration_file = model_file_download(
                model_id=model_id,
                file_path=ModelFile.CONFIGURATION,
                revision=model_revision)
        cfg = Config.from_file(configuration_file)
        print(f"cfg is {cfg}")
        task = cfg.safe_get('task')
        print(f"task is {task}, model is {model_id}, model_revision is {model_revision}")

        # init pipeline
        ppl = pipeline(
            task=task,
            model=model_id,
            model_revision=model_revision,
            llm_first=True)

        print(f"ppl is {ppl}")
        pipeline_info = get_pipeline_information_by_pipeline(ppl)
        print(f"pipeline_info is {pipeline_info}")

        # call pipeline
        data = get_task_input_examples(task)
        print(f"data is {data}")

        infer_result = call_pipeline_with_json(pipeline_info, ppl, data)
        result = pipeline_output_to_service_base64_output(task, infer_result)
        return result


if __name__ == '__main__':
    # task_model_map = {
    #     "text-to-image-synthesis": [
    #         "AI-ModelScope/stable-diffusion-v2-1"
    #     ],
    #     "face-detection": [
    #         "damo/cv_resnet50_face-detection_retinaface"
    #     ],
    #     "face-recognition": [
    #         "damo/cv_ir_face-recognition-ood_rts"
    #     ],
    #     "vision-efficient-tuning": [
    #         "damo/cv_vitb16_classification_vision-efficient-tuning-lora"
    #     ],
    #     "video-single-object-tracking": [
    #         "damo/cv_alex_video-single-object-tracking_siamfc-uav"
    #     ],
    #     "ocr-detection": [
    #         "damo/cv_resnet18_ocr-detection-db-line-level_damo"
    #     ]
    # }
    model_list = [
        # 'qwen/Qwen-7B-Chat-Int4',
        # 'qwen/Qwen-14B-Chat-Int4',
        # 'baichuan-inc/Baichuan2-7B-Chat-4bits',
        # 'baichuan-inc/Baichuan2-13B-Chat-4bits',
        # 'ZhipuAI/chatglm2-6b-int4',
        # 'AI-ModelScope/stable-diffusion-v2-1',
        # 'ZhipuAI/chatglm2-6b',

        # face-detection
        'damo/cv_manual_face-detection_tinymog',
        'damo/cv_ddsar_face-detection_iclr23-damofd',
        'damo/cv_ddsar_face-detection_iclr23-damofd-34G',
        'damo/cv_ddsar_face-detection_iclr23-damofd-2.5G',
        'damo/cv_ddsar_face-detection_iclr23-damofd-10G',
        'damo/cv_manual_uav-detection_uav',
        'damo/cv_resnet50_face-detection_retinaface',
        'damo/cv_resnet_facedetection_scrfd10gkps',
        'damo/cv_resnet101_face-detection_cvpr22papermogface',
        'damo/cv_manual_face-detection_mtcnn',
        'damo/cv_manual_face-detection_ulfd',

        # face-recognition
        'damo/cv_ir_face-recognition-ood_rts',
        'damo/cv_vit_face-recognition',
        'damo/cv_manual_face-recognition_frir',
        'damo/cv_manual_face-recognition_frfm',
        'damo/cv_ir101_facerecognition_cfglint',
        'damo/cv_ir50_face-recognition_arcface',
        'damo/cv_resnet_face-recognition_facemask',

        # vision-efficient-tuning
        'damo/cv_vitb16_classification_vision-efficient-tuning-utuning',
        'damo/cv_vitb16_classification_vision-efficient-tuning-lora',
        'damo/cv_vitb16_classification_vision-efficient-tuning-prompt',
        'damo/cv_vitb16_classification_vision-efficient-tuning-adapter',
        'damo/cv_vitb16_classification_vision-efficient-tuning-prefix',
        'damo/cv_vitb16_classification_vision-efficient-tuning-sidetuning',
        'damo/cv_vitb16_classification_vision-efficient-tuning-bitfit',
        'damo/cv_vitb16_classification_vision-efficient-tuning-base',

        # ocr-detection
        'damo/cv_resnet18_ocr-detection-db-line-level_damo',
        'damo/cv_resnet18_ocr-detection-line-level_damo',
        'damo/cv_resnet18_ocr-detection-word-level_damo',
        'damo/cv_proxylessnas_ocr-detection-db-line-level_damo',
        'damo/cv_resnet50_ocr-detection-vlpt',

        # video-single-object-tracking
        'damo/cv_vitb_video-single-object-tracking_ostrack-uav-l',
        'damo/cv_vitb_video-single-object-tracking_ostrack-l',
        'damo/cv_alex_video-single-object-tracking_siamfc-uav',
        'damo/cv_alex_video-single-object-tracking_siamfc-uav',
        'damo/cv_vitb_video-single-object-tracking_procontext',
        'damo/cv_vitb_video-single-object-tracking_ostrack',

        # face-reconstruction
        'damo/cv_resnet50_face-reconstruction',

        # image-inpainting
        'damo/cv_stable-diffusion-v2_image-inpainting_base',
        'damo/cv_stable-diffusion-v2_image-inpainting_base',
        'damo/cv_fft_inpainting_lama',

        # text-to-image-synthesis
        'langboat/Guohua-Diffusion',
        'AI-ModelScope/stable-diffusion-xl-base-1.0',
        'Fengshenbang/Taiyi-Stable-Diffusion-1B-Chinese-v0.1',
        'Fengshenbang/Taiyi-Stable-Diffusion-1B-Anime-Chinese-v0.1',
        'AI-ModelScope/stable-diffusion-v1-5',
        'damo/cv_cartoon_stable_diffusion_design',
        'Fengshenbang/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1',
        'damo/cv_diffusion_text-to-image-synthesis_tiny',
        'damo/cv_cartoon_stable_diffusion_clipart',
        'damo/cv_cartoon_stable_diffusion_illustration',
        'AI-ModelScope/rwkv-4-world',
        'PAI/pai-diffusion-artist-xlarge-zh',
        'AI-ModelScope/stable-diffusion-v1.5-no-safetensor',
        'PAI/pai-diffusion-artist-large-zh',
        'damo/cv_cartoon_stable_diffusion_watercolor',
        'damo/cv_cartoon_stable_diffusion_flat',
        'AI-ModelScope/stable-diffusion-v2-1',
        'AI-ModelScope/stable-diffusion-xl-refiner-1.0',
        'modelscope/small-stable-diffusion-v0',
        'AI-ModelScope/t5-base',
        'damo/cv_composer_multi-modal-image-synthesis',
        'WordArt/font_generation_base_model',
        'AI-ModelScope/stable-diffusion-2-1',
        'AI-ModelScope/stable-diffusion-v1.4',
        'AI-ModelScope/stable-diffusion-GhostMix-V1-1',
        'AI-ModelScope/rwkv-4-music',
        'AI-ModelScope/stable-diffusion-GhostMix-V1-2-fp16-pruned',
        'PAI/pai-diffusion-anime-large-zh',
        'AI-ModelScope/Realistic_Vision_V4.0',
        'AI-ModelScope/stable-diffusion-v1-4',
        'PAI/pai-diffusion-general-xlarge-zh',
        'modelscope/falcon-180B',
        'PAI/pai-diffusion-food-large-zh',
        'PAI/pai-diffusion-general-large-zh',
        'AI-ModelScope/stable-diffusion-xl-base-0.9',
        'damo/multi-modal_chinese_stable_diffusion_v1.0',
        'damo/ofa_text-to-image-synthesis_coco_large_en',

        # image-skychange
        'damo/cv_hrnetocr_skychange',

        # image-deblurring
        'damo/cv_nafnet_image-deblur_gopro',
        'damo/cv_nafnet_image-deblur_reds',

        # video-frame-interpolation
        'damo/cv_raft_video-frame-interpolation_practical',
        'damo/cv_raft_video-frame-interpolation',

        # image-quality-assessment-mos
        'damo/cv_resnet_image-quality-assessment-mos_youtubeUGC',
        'damo/cv_man_image-quality-assessment',

        # image-driving-perception
        'damo/cv_yolopv2_image-driving-perception_bdd100k'

        # image-quality-assessment-degradation
        'damo/cv_resnet50_image-quality-assessment_degradation'

        # bad-image-detecting
        'damo/cv_mobilenet-v2_bad-image-detecting',

        # video-object-detection
        'damo/cv_cspnet_video-object-detection_longshortnet',
        'damo/cv_cspnet_video-object-detection_streamyolo',

        # video-deinterlace
        'damo/cv_unet_video-deinterlace',
        
        # image-to-video
        'damo/Image-to-Video',
        
        # video-to-video
        'damo/Video-to-Video',

        # face-quality-assessment
        'damo/cv_manual_face-quality-assessment_fqa',
        
        # human-reconstruction
        'damo/cv_hrnet_image-human-reconstruction',

        # image-paintbyexample
        'damo/cv_stable-diffusion_paint-by-example',

        # image_variation_task
        'damo/cv_image_variation_sd',

        # video-super-resolution
        'damo/cv_msrresnet_video-super-resolution_lite',
        'damo/cv_realbasicvsr_video-super-resolution_videolq',

        # image-super-resolution
        # image-super-resolution-pasd
        'bilibili/cv_bilibili_image-super-resolution',
        'damo/PASD_v2_image_super_resolutions',
        'damo/cv_ecbsr_image-super-resolution_mobile',
        'damo/PASD_image_super_resolutions',
        'damo/cv_rrdb_image-super-resolution',

        # head-reconstruction
        'damo/cv_HRN_head-reconstruction',

        # face_fusion_torch
        'damo/cv_unet_face_fusion_torch',

        # video-temporal-grounding
        # soonet_video_temporal_grounding_test_video.mp4
        'damo/multi-modal_soonet_video-temporal-grounding',
        
        # image-try-on
        'damo/cv_SAL-VTON_virtual-try-on',

        # video-text-retrieval
        'damo/cv_vit-b32_retrieval_vop_bias',
        'damo/cv_vit-b32_retrieval_vop_partial',
        'damo/cv_vit-b32_retrieval_vop_proj',
        'damo/cv_vit-b32_retrieval_vop',

        # face-liveness
        'damo/cv_manual_face-liveness_flxc',
        'damo/cv_manual_face-liveness_flrgb',
        'damo/cv_manual_face-liveness_flir',

        # image-color-enhancement
        'damo/cv_adaint_image-color-enhance-models',
        'damo/cv_deeplpfnet_image-color-enhance-models',
        'damo/cv_csrnet_image-color-enhance-models',

        # video-object-segmentation
        'damo/cv_rdevos_video-object-segmentation',
        'damo/cv_mivos-stcn_video-object-segmentation',

    ]
    tester = ModelJsonTest()
    for model in model_list:
        try:
            print(f"model is {model}")
            res = tester.test_single(model)
            print(
                f'\nmodel_id {model} call_pipeline_with_json run ok. {res}\n\n\n\n'
            )
        except BaseException as e:
            print(
                f'\nmodel_id {model} call_pipeline_with_json run failed: {e}.\n\n\n\n'
            )
