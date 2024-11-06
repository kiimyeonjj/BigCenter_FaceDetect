import cv2
import numpy as np
from multiprocessing import Pool, cpu_count, Value, Lock
from typing import List, Tuple
from tqdm import tqdm
import os

# 공유 진행 상황을 위한 전역 변수
progress_counter = None
total_frames = None

def init_progress(progress, frames):
    global progress_counter, total_frames
    progress_counter = progress
    total_frames = frames

def process_video_chunk(chunk_info: Tuple[str, int, int]) -> List[np.ndarray]:
    video_path, start_frame, end_frame = chunk_info
    
    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 비디오를 열 수 없습니다 - {video_path}")
        return []
    
    # 시작 프레임으로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    current_frame = start_frame
    chunk_size = end_frame - start_frame
    
    # 프로그레스바 설정
    pbar = tqdm(total=chunk_size, 
                desc=f'Process {os.getpid()} ({start_frame}-{end_frame})', 
                position=start_frame // max(1, chunk_size))
    
    try:
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: 프레임 {current_frame} 읽기 실패")
                break
                
            # 여기에 나중에 딥러닝 추론 코드가 들어갈 예정
            processed_frame = frame.copy()  # 깊은 복사 수행
            
            frames.append(processed_frame)
            current_frame += 1
            
            # 진행 상황 업데이트
            with progress_counter.get_lock():
                progress_counter.value += 1
            pbar.update(1)
            
    except Exception as e:
        print(f"Error in process_video_chunk: {e}")
        
    finally:
        pbar.close()
        cap.release()
        
    print(f"Process {os.getpid()} completed: {len(frames)} frames processed")
    return frames

def process_video_parallel(video_path: str, output_path: str = None) -> None:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
    
    # 입력 비디오 정보 획득
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"비디오를 열 수 없습니다: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    
    print(f"비디오 정보: {total_frames} frames, {fps} fps, {width}x{height}")
    
    if total_frames <= 0:
        raise ValueError("비디오에 프레임이 없습니다.")
    
    # 출력 경로 설정
    if output_path is None:
        filename, ext = os.path.splitext(video_path)
        output_path = f"{filename}_processed{ext}"
    
    # 진행 상황 공유 변수 초기화
    progress = Value('i', 0)
    
    # 코어 수 설정 및 청크 계산
    num_cores = min(cpu_count(), total_frames)
    frames_per_core = max(1, total_frames // num_cores)
    
    # 청크 정보 생성
    chunks = []
    for i in range(num_cores):
        start_frame = i * frames_per_core
        end_frame = min(start_frame + frames_per_core, total_frames)
        chunks.append((video_path, start_frame, end_frame))
    
    print(f"Processing with {num_cores} cores, {frames_per_core} frames per core")
    
    # 전체 진행 상황 표시 설정
    main_pbar = tqdm(total=total_frames, desc='Total Progress', position=num_cores+1)
    
    # 병렬 처리 실행
    with Pool(processes=num_cores, initializer=init_progress, 
             initargs=(progress, total_frames)) as pool:
        results = pool.map(process_video_chunk, chunks)
    
    main_pbar.close()
    
    # 결과 합치기
    processed_frames = []
    for chunk_result in results:
        processed_frames.extend(chunk_result)
    
    print(f"총 처리된 프레임: {len(processed_frames)}")
    
    if not processed_frames:
        raise ValueError("처리된 프레임이 없습니다!")
    
    # 결과 저장
    print(f"\n저장 중: {output_path}")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"출력 비디오 파일을 생성할 수 없습니다: {output_path}")
    
    for frame in tqdm(processed_frames, desc="Saving video"):
        out.write(frame)
    
    out.release()
    print(f"처리 완료! 저장된 파일: {output_path}")
