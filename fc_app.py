import os
import sys
import subprocess
import time
import traceback

from flask import Flask, request


# TODO rwlock pipe
FC_MODEL_CACHE_DIR = os.getenv('MODELSCOPE_CACHE', '/mnt/auto/')
ut_file_name = os.getenv('UTFILE', '')

# get model-info form env 

app = Flask(__name__)

### XXX modelscope end

@app.route('/', methods=['POST'])
@app.route('/invoke', methods=['POST'])
def invoke():
    # See FC docs for all the HTTP headers: https://help.aliyun.com/document_detail/179368.html#section-fk2-z5x-am6
    request_id = request.headers.get("x-fc-request-id", "")
    print("[INFO] modelscope-ut-on-fc Invoke Start RequestId: " + request_id)

    try:
        elapsed = 0
        global ut_file_name
        if ut_file_name is None:
            stime = time.time()
            cmd = f'python tests/pipelines/{ut_file_name}'
            print("[INFO] ut test cmd: " + cmd)
            result = subprocess.call(cmd, shell=True)
            if result == 0:
                print("单测执行成功")
                return {
                           'Code': 200,
                           'Message': "单测执行成功",
                           'Data': "",
                           "RequestId": request_id,
                           "Success": True
                       }, 200, [("Content-Type", "application/json")]
            else:
                print("单测执行失败，错误码：", result)
                return {
                           'Code': 500,
                           'Message': f"单测执行失败，错误码：{result}",
                           'Data': "",
                           "RequestId": request_id,
                           "Success": True
                       }, 200, [("Content-Type", "application/json")]

            etime = time.time()
            elapsed = etime - stime
        print(f"[INFO] invoke elapsed: {elapsed:.1f}")
    except Exception as e:
        print("[ERROR] modelscope-ut-on-fc Invoke: " + str(e))
        exc_info = sys.exc_info()
        trace = traceback.format_tb(exc_info[2])
        err_ret = {
            'Code': 400,
            'Message': str(trace),
            'Data': "",
            "RequestId": request_id,
            "Success": False
        }
        print("[INFO] modelscope-ut-on-fc Invoke End RequestId: " + request_id)
        return err_ret, 400, [("x-fc-status", "400")]


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=9000)
