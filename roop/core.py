#!/usr/bin/env python3

import os
import sys
import shutil
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'

import warnings
from typing import List
import platform
import signal
import torch
import onnxruntime
import pathlib
import argparse

from time import time

import roop.globals
import roop.metadata
import roop.utilities as util
import roop.util_ffmpeg as ffmpeg
import ui.main as main
from settings import Settings
from roop.face_util import extract_face_images
from roop.ProcessEntry import ProcessEntry
from roop.ProcessMgr import ProcessMgr
from roop.ProcessOptions import ProcessOptions
from roop.capturer import get_video_frame_total, release_video


clip_text = None

call_display_ui = None

process_mgr = None


if 'ROCMExecutionProvider' in roop.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    roop.globals.headless = False

    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('--server_share', help='Public server', dest='server_share', action='store_true', default=False)
    program.add_argument('--cuda_device_id', help='Index of the cuda gpu to use', dest='cuda_device_id', type=int, default=0)
    roop.globals.startup_args = program.parse_args()
    # Always enable all processors when using GUI
    roop.globals.frame_processors = ['face_swapper', 'face_enhancer']


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    list_providers = [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]
    
    try:
        for i in range(len(list_providers)):
            if list_providers[i] == 'CUDAExecutionProvider':
                list_providers[i] = ('CUDAExecutionProvider', {'device_id': roop.globals.cuda_device_id})
                torch.cuda.set_device(roop.globals.cuda_device_id)
                break
    except:
        pass

    return list_providers
    


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in roop.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in roop.globals.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    # limit memory usage
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = roop.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))



def release_resources() -> None:
    import gc
    global process_mgr

    if process_mgr is not None:
        process_mgr.release_resources()
        process_mgr = None

    gc.collect()
    # if 'CUDAExecutionProvider' in roop.globals.execution_providers and torch.cuda.is_available():
    #     with torch.cuda.device('cuda'):
    #         torch.cuda.empty_cache()
    #         torch.cuda.ipc_collect()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    
    download_directory_path = util.resolve_relative_path('../models')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/GFPGANv1.4.onnx'])
    util.conditional_download(download_directory_path, ['https://github.com/csxmli2016/DMDNet/releases/download/v1/DMDNet.pth'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/GPEN-BFR-512.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/restoreformer_plus_plus.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/xseg.onnx'])
    download_directory_path = util.resolve_relative_path('../models/CLIP')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/rd64-uni-refined.pth'])
    download_directory_path = util.resolve_relative_path('../models/CodeFormer')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/CodeFormerv0.1.onnx'])
    download_directory_path = util.resolve_relative_path('../models/Frame')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/deoldify_artistic.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/deoldify_stable.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/isnet-general-use.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/real_esrgan_x4.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/real_esrgan_x2.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/lsdir_x4.onnx'])

    if not shutil.which('ffmpeg'):
       update_status('ffmpeg is not installed.')
    return True

def set_display_ui(function):
    global call_display_ui

    call_display_ui = function


def update_status(message: str) -> None:
    global call_display_ui

    print(message)
    if call_display_ui is not None:
        call_display_ui(message)


def start() -> None:
    if roop.globals.headless:
        print('Headless mode currently unsupported - starting UI!')
        # faces = extract_face_images(roop.globals.source_path,  (False, 0))
        # roop.globals.INPUT_FACES.append(faces[roop.globals.source_face_index])
        # faces = extract_face_images(roop.globals.target_path,  (False, util.has_image_extension(roop.globals.target_path)))
        # roop.globals.TARGET_FACES.append(faces[roop.globals.target_face_index])
        # if 'face_enhancer' in roop.globals.frame_processors:
        #     roop.globals.selected_enhancer = 'GFPGAN'
       
    batch_process_regular(None, False, None)


def get_processing_plugins(masking_engine):
    processors = {  "faceswap": {}}
    if masking_engine is not None:
        processors.update({masking_engine: {}})
    
    if roop.globals.selected_enhancer == 'GFPGAN':
        processors.update({"gfpgan": {}})
    elif roop.globals.selected_enhancer == 'Codeformer':
        processors.update({"codeformer": {}})
    elif roop.globals.selected_enhancer == 'DMDNet':
        processors.update({"dmdnet": {}})
    elif roop.globals.selected_enhancer == 'GPEN':
        processors.update({"gpen": {}})
    elif roop.globals.selected_enhancer == 'Restoreformer++':
        processors.update({"restoreformer++": {}})
    return processors


def live_swap(frame, options):
    global process_mgr

    if frame is None:
        return frame

    if process_mgr is None:
        process_mgr = ProcessMgr(None)
    
#    if len(roop.globals.INPUT_FACESETS) <= selected_index:
#        selected_index = 0
    process_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, options)
    new_frame = process_mgr.process_frame(frame)
    return new_frame


def batch_process_regular(options, preview, mask_engine):
    global process_mgr
    if process_mgr is None:
        process_mgr = ProcessMgr(options)

    new_frame = None

    process_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, options)
    
    # Process all frames here
    # for frame in roop.globals.input_video_reader:
    #     new_frame = process_mgr.process_frame(frame)
    
    # if preview:
    #     return new_frame
    
    process_mgr.release_resources()


def destroy() -> None:
    print('Application terminated!')
    release_resources()
    exit(0)


def run() -> None:
    parse_args()
    if not pre_check():
        return
    roop.globals.CFG = Settings('config.yaml')
    roop.globals.cuda_device_id = roop.globals.startup_args.cuda_device_id
    roop.globals.execution_threads = roop.globals.CFG.max_threads
    roop.globals.video_encoder = roop.globals.CFG.output_video_codec
    roop.globals.video_quality = roop.globals.CFG.video_quality
    roop.globals.max_memory = roop.globals.CFG.memory_limit if roop.globals.CFG.memory_limit > 0 else None
    
    # Check if server_share is enabled and pass it to main.run()
    server_share = roop.globals.startup_args.server_share
    main.run(share=server_share)  # Pass the share argument
    
