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
        'damo/cv_resnet50_face-detection_retinaface',
        # 'damo/cv_ir_face-recognition-ood_rts',
        # 'damo/cv_vitb16_classification_vision-efficient-tuning-lora',
        # 'damo/cv_alex_video-single-object-tracking_siamfc-uav',
        # 'damo/cv_resnet18_ocr-detection-db-line-level_damo'
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
