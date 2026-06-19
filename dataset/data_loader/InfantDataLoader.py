import os
import glob
import cv2
import numpy as np
import pandas as pd
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import csv

class InfantDataLoader(BaseLoader):
    """
    Custom loader for reading `.bmp` image sequences nested inside a 'Cam 1' folder 
    within a participant directory, and `.txt` pulse wave labels.
    """
    def __init__(self, dataset_name, raw_data_path, label_data_path, config_data, device=None):
        self.label_data_path = label_data_path
        super().__init__(dataset_name, raw_data_path, config_data, device)
        self.stats_csv_path = self.config_data.STATS_CSV.PATH
        self.stats_available = self.config_data.STATS_CSV.AVAILABLE
        self.participant_stats = {}

        if self.do_preprocess or not self.stats_available:
            print("Generating participant-wise statistics...")
            self.generate_participant_stats()
        
        self.load_participant_stats()

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
    
    def generate_participant_stats(self):
        """
        Iterates through raw chunks to calculate global mean/std for Standardized mode,
        and the global standard deviation of differences for DiffNormalized mode.
        """
        raw_stats = {}
        
        # Step 1: Accumulate raw properties and temporal differences
        for input_path in self.inputs:
            filename = os.path.basename(input_path)
            participant_id = filename.split('_')[0]
            
            data = np.load(input_path)
            label_path = input_path.replace("input", "label")
            label = np.load(label_path)
            
            if participant_id not in raw_stats:
                raw_stats[participant_id] = {
                    'data_sum': 0.0, 'data_sq_sum': 0.0, 'data_count': 0,
                    'label_sum': 0.0, 'label_sq_sum': 0.0, 'label_count': 0,
                    # Accumulators for temporal difference variations
                    'diff_data_sq_sum': 0.0, 'diff_data_count': 0,
                    'diff_label_sq_sum': 0.0, 'diff_label_count': 0
                }
            
            # Standard metrics accumulation
            raw_stats[participant_id]['data_sum'] += np.sum(data)
            raw_stats[participant_id]['data_sq_sum'] += np.sum(np.square(data))
            raw_stats[participant_id]['data_count'] += data.size
            
            raw_stats[participant_id]['label_sum'] += np.sum(label)
            raw_stats[participant_id]['label_sq_sum'] += np.sum(np.square(label))
            raw_stats[participant_id]['label_count'] += label.size

            # Diff metrics calculation (replicating BaseLoader math formulas)
            # Data diff formula: (next - curr) / (next + curr + 1e-7)
            if data.shape[0] > 1:
                diff_data = (data[1:] - data[:-1]) / (data[1:] + data[:-1] + 1e-7)
                raw_stats[participant_id]['diff_data_sq_sum'] += np.sum(np.square(diff_data))
                raw_stats[participant_id]['diff_data_count'] += diff_data.size

            # Label diff formula: np.diff(label)
            if label.shape[0] > 1:
                diff_label = np.diff(label, axis=0)
                raw_stats[participant_id]['diff_label_sq_sum'] += np.sum(np.square(diff_label))
                raw_stats[participant_id]['diff_label_count'] += diff_label.size

        # Step 2: Finalize calculations and output CSV
        os.makedirs(os.path.dirname(self.stats_csv_path), exist_ok=True)
        with open(self.stats_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'participant_id', 
                'data_mean', 'data_std', 'diff_data_std',
                'label_mean', 'label_std', 'diff_label_std'
            ])
            
            for pid, stats in raw_stats.items():
                # Standard calculations
                d_mean = stats['data_sum'] / stats['data_count']
                d_std = np.sqrt(max((stats['data_sq_sum'] / stats['data_count']) - np.square(d_mean), 1e-7))
                
                l_mean = stats['label_sum'] / stats['label_count']
                l_std = np.sqrt(max((stats['label_sq_sum'] / stats['label_count']) - np.square(l_mean), 1e-7))
                
                # Diff calculations (Mean of differences is assumed centered around 0 in standard pipelines)
                diff_d_std = np.sqrt(max(stats['diff_data_sq_sum'] / stats['diff_data_count'], 1e-7))
                diff_l_std = np.sqrt(max(stats['diff_label_sq_sum'] / stats['diff_label_count'], 1e-7))
                
                writer.writerow([pid, d_mean, d_std, diff_d_std, l_mean, l_std, diff_l_std])
                
        print(f"Participant metrics successfully written to: {self.stats_csv_path}")

    def load_participant_stats(self):
        df = pd.read_csv(self.stats_csv_path, dtype={'participant_id': str})
        for _, row in df.iterrows():
            self.participant_stats[str(row['participant_id'])] = {
                'data_mean': row['data_mean'], 'data_std': row['data_std'], 'diff_data_std': row['diff_data_std'],
                'label_mean': row['label_mean'], 'label_std': row['label_std'], 'diff_label_std': row['diff_label_std']
            }

    def __getitem__(self, index):
        data = np.load(self.inputs[index])
        label = np.load(self.labels[index])
        
        item_path = self.inputs[index]
        item_path_filename = item_path.split(os.sep)[-1]
        participant_id = item_path_filename.split('_')[0]
        
        stats = self.participant_stats[participant_id]

        # 2. Dynamic Video Data Scaling
        scaled_data_list = []
        for data_type in self.config_data.PREPROCESS.DATA_TYPE:
            f_c = data.copy()
            if data_type == "Raw":
                scaled_data_list.append(f_c)
                
            elif data_type == "Standardized":
                standardized = (f_c - stats['data_mean']) / stats['data_std']
                standardized[np.isnan(standardized)] = 0
                scaled_data_list.append(standardized)
                
            elif data_type == "DiffNormalized":
                n, h, w, c = f_c.shape
                
                # Vectorized calculation using NumPy slicing
                diff_data = (f_c[1:] - f_c[:-1]) / (f_c[1:] + f_c[:-1] + 1e-7)
                
                # Normalize via full-video diff standard deviation profile mapping
                diff_data = diff_data / stats['diff_data_std']
                
                # Pad with a single zero-frame at the end to keep chunk shapes aligned (N frames)
                diff_data = np.append(diff_data, np.zeros((1, h, w, c), dtype=np.float32), axis=0)
                diff_data[np.isnan(diff_data)] = 0
                scaled_data_list.append(diff_data)
                
            else:
                raise ValueError(f"Unsupported dynamic data scaling type: {data_type}")
        data = np.concatenate(scaled_data_list, axis=-1)

        # 3. Dynamic Label Signal Scaling
        label_type = self.config_data.PREPROCESS.LABEL_TYPE
        if label_type == "Raw":
            pass
            
        elif label_type == "Standardized":
            label = (label - stats['label_mean']) / stats['label_std']
            label[np.isnan(label)] = 0
            
        elif label_type == "DiffNormalized":
            diff_label = np.diff(label, axis=0)
            # Normalize via full-video diff standard deviation profile mapping
            diff_label = diff_label / stats['diff_label_std']
            # Pad to keep chunk shapes aligned
            label = np.append(diff_label, np.zeros(1), axis=0)
            label[np.isnan(label)] = 0
            
        else:
            raise ValueError(f"Unsupported dynamic label scaling type: {label_type}")

        # 4. Dimension transpositions
        if self.data_format == 'NDCHW':
            data = np.transpose(data, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            data = np.transpose(data, (3, 0, 1, 2))
            
        data = np.float32(data)
        label = np.float32(label)
        
        split_idx = item_path_filename.rindex('_')
        filename = item_path_filename[:split_idx]
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]
        
        return data, label, filename, chunk_id
    
    @property
    def groups(self):
        """
        Extracts and returns the participant IDs from the preprocessed filenames
        to serve as the 'groups' for GroupKFold.
        """
        # BaseLoader saves files as: {participant_id}_input{chunk_id}.npy
        # We split the filename to grab just the participant ID
        return [os.path.basename(inp).split('_')[0] for inp in self.inputs]