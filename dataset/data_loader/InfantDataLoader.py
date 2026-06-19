import os
import glob
import cv2
import numpy as np
import pandas as pd
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm

class InfantDataLoader(BaseLoader):
    """
    Custom loader for reading `.bmp` image sequences nested inside a 'Cam 1' folder 
    within a participant directory, and `.txt` pulse wave labels.
    """
    def __init__(self, dataset_name, raw_data_path, label_data_path, config_data, device=None):
        self.label_data_path = label_data_path
        super().__init__(dataset_name, raw_data_path, config_data, device)

    def get_raw_data(self, raw_data_path):
        """
        Locates directories based on the folder structure.
        Expected structure: raw_data_path / *[participant_id] / Cam 1 / *.bmp
        """
        data_dirs = []
        participant_folders = [f.path for f in os.scandir(raw_data_path) if f.is_dir()]
        
        for p_folder in participant_folders:
            folder_name = os.path.basename(p_folder)
            participant_id = folder_name[-2:]
            
            cam_1_folder = os.path.join(p_folder, "Cam 1")
            label_path = os.path.join(self.label_data_path, f"{participant_id}.txt")
            
            if not os.path.exists(label_path):
                continue
            
            if not os.path.exists(cam_1_folder):
                continue
            
            if not glob.glob(os.path.join(cam_1_folder, '*.bmp')):
                continue
                
            data_dirs.append({
                'index': participant_id,
                'path': cam_1_folder
            })
            
        return data_dirs

    def split_raw_data(self, data_dirs, begin, end):
        # total = len(data_dirs)
        # start_idx = int(round(begin * total))
        # end_idx = int(round(end * total))
        return data_dirs #[start_idx:end_idx]

    def read_bmp_video(self, image_dir):
        image_paths = sorted(glob.glob(os.path.join(image_dir, '*.bmp')))
        
        frames = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img)
                
        return np.asarray(frames)

    def read_pulse_wave(self, participant_id):
        """Reads the tab-separated .txt pulse wave label, keeping only the first column."""
        label_path = os.path.join(self.label_data_path, f"{participant_id}.txt")
        pulse_wave = np.loadtxt(label_path, delimiter='\t', usecols=1) 
        return pulse_wave

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """
        Handles reading, resampling using BaseLoader.resample_ppg, truncating to 
        the shorter of the two modalities, and passing to the BaseLoader core.
        """
        data_info = data_dirs[i]
        participant_id = data_info['index']
        img_dir = data_info['path']

        # 1. Read Raw Data
        frames = self.read_bmp_video(img_dir)
        bvps = self.read_pulse_wave(participant_id)

        # 2. Extract FPS and Sampling Rate from the configuration metadata
        video_fps = float(self.config_data.VIDEO_FPS)
        signal_fs = float(self.config_data.SIGNAL_FS)

        # 3. Resample the signal down to match the video target framerate
        target_signal_length = int(len(bvps) * (video_fps / signal_fs))
        bvps_resampled = BaseLoader.resample_ppg(bvps, target_signal_length)

        # 4. Truncate to the SHORTER of the two lengths to ensure perfect alignment
        min_length = min(len(frames), len(bvps_resampled))
        frames = frames[:min_length]
        bvps_resampled = bvps_resampled[:min_length]

        # 5. Pass to BaseLoader's core preprocessing (face crop, resize, chunking)
        frames_clips, bvps_clips = self.preprocess(frames, bvps_resampled, config_preprocess)

        # 6. Save chunked data
        input_paths, label_paths = self.save_multi_process(frames_clips, bvps_clips, participant_id)
        file_list_dict[i] = input_paths
        
    def load_preprocessed_data(self):
        """Loads all preprocessed data into memory/lists."""
        file_list_df = pd.read_csv(self.file_list_path)
        self.inputs = sorted(file_list_df['input_files'].tolist())
        self.labels = [inp.replace("input", "label") for inp in self.inputs]
        self.preprocessed_data_len = len(self.inputs)

    def multi_process_manager(self, data_dirs, config_preprocess):
        """Allocate dataset preprocessing sequentially.

        Args:
            data_dirs(List[str]): a list of video_files.
            config_preprocess(Dict): a dictionary of preprocessing configurations
        Returns:
            file_list_dict(Dict): Dictionary containing information regarding processed data ( path names)
        """
        print('Preprocessing dataset...')
        file_num = len(data_dirs)

        # Standard standard Python dictionary instead of mp.Manager().dict()
        file_list_dict = {}

        # Iterate sequentially with a simplified tqdm progress bar
        for i in tqdm(range(file_num)):
            # Call the preprocessing function directly in the main thread
            self.preprocess_dataset_subprocess(
                data_dirs, 
                config_preprocess, 
                i, 
                file_list_dict
            )

        return file_list_dict
    
    @property
    def groups(self):
        """
        Extracts and returns the participant IDs from the preprocessed filenames
        to serve as the 'groups' for GroupKFold.
        """
        # BaseLoader saves files as: {participant_id}_input{chunk_id}.npy
        # We split the filename to grab just the participant ID
        return [os.path.basename(inp).split('_')[0] for inp in self.inputs]
    
    def __getitem__(self, index):
        """
        Returns a clip of video and its corresponding signals, applying 
        the scaling/normalization specified in config_data on the fly.
        """
        data = np.load(self.inputs[index])
        label = np.load(self.labels[index])

        scaled_data_list = []
        for data_type in self.config_data.PREPROCESS.DATA_TYPE:
            f_c = data.copy()
            if data_type == "Raw":
                scaled_data_list.append(f_c)
            elif data_type == "DiffNormalized":
                scaled_data_list.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                scaled_data_list.append(BaseLoader.standardized_data(f_c))
            else:
                raise ValueError(f"Unsupported dynamic data scaling type: {data_type}")
        
        data = np.concatenate(scaled_data_list, axis=-1)

        label_type = self.config_data.PREPROCESS.LABEL_TYPE
        if label_type == "Raw":
            pass
        elif label_type == "DiffNormalized":
            label = BaseLoader.diff_normalize_label(label)
        elif label_type == "Standardized":
            label = BaseLoader.standardized_label(label)
        else:
            raise ValueError(f"Unsupported dynamic label scaling type: {label_type}")

        if self.data_format == 'NDCHW':
            data = np.transpose(data, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            data = np.transpose(data, (3, 0, 1, 2))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')
            
        data = np.float32(data)
        label = np.float32(label)
        
        item_path = self.inputs[index]
        item_path_filename = item_path.split(os.sep)[-1]
        split_idx = item_path_filename.rindex('_')
        filename = item_path_filename[:split_idx]
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]
        
        return data, label, filename, chunk_id