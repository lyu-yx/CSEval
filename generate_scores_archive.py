
import os
import glob
import json
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from scipy.stats import spearmanr
from scipy.fft import fft2, fftshift
from skimage.measure import shannon_entropy
from skimage.feature import hog
from scipy.ndimage import gaussian_laplace
import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
from skimage.feature import canny, local_binary_pattern
from sklearn.isotonic import IsotonicRegression
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import tqdm

# Setup: Seeds and constants
np.random.seed(42)
LBP_RADIUS = 1
LBP_N_POINTS = 8
BG_DILATION_KERNEL = np.ones((17, 17), np.uint8) # 8px dilation = (2*8+1) kernel size

def main():
    """
    Main function to run the entire scoring pipeline.
    """
    # Define paths
    dataset_root = Path("/HDD_16T/rsy/UEDG-master/dataset/")
    output_dir = Path("./")
    output_dir.mkdir(exist_ok=True)

    # Output file paths
    csv_path = output_dir / "gt_continuous.csv"
    features_cache_path = output_dir / "features_cache.pkl"
    ecdf_params_path = output_dir / "ecdf_params.json"
    isotonic_cod_path = output_dir / "isotonic_cod.pkl"
    isotonic_sod_path = output_dir / "isotonic_sod.pkl"
    
    # Run pipeline
    print("--- Starting Data Processing Pipeline ---")

    # Step 0 & 1: Enumerate files and get metadata
    all_files = get_file_list(dataset_root)
    if not all_files:
        print("No image/mask files found. Please check dataset paths.")
        return
    df_files = pd.DataFrame(all_files)
    print(f"Found {len(df_files)} total valid mask files.")

    # Step 2: Compute features
    if features_cache_path.exists():
        print(f"Loading features from cache: {features_cache_path}")
        features_df = pd.read_pickle(features_cache_path)
    else:
        print("Computing features for all samples using multiple processes...")
        
        # Use a sensible number of worker processes
        num_workers = max(1, int(multiprocessing.cpu_count() * 0.8))
        print(f"Using {num_workers} worker processes.")

        features_list = []
        tasks = list(zip(df_files['img_path'], df_files['mask_path']))

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks to the pool
            future_to_task = {executor.submit(compute_all_features, img_path, mask_path): (img_path, mask_path) for img_path, mask_path in tasks}
            
            # Process results as they complete to show progress
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Computing features"):
                result = future.result()
                if result:
                    features_list.append(result)
                
                # Log progress periodically
                if (len(features_list) % 10000 == 0) and (len(features_list) > 0):
                    # Use tqdm.write to avoid interfering with the progress bar
                    tqdm.write(f"Processed {len(features_list)} samples.")
        
        if not features_list:
            print("No features were computed. Exiting.")
            return

        features_df = pd.DataFrame(features_list)
        features_df.to_pickle(features_cache_path)
        print(f"Saved features to cache: {features_cache_path}")

    # Merge features with file info
    data_df = pd.merge(df_files, features_df, on=['img_path', 'mask_path'])

    # Step 3: ECDF transformation for ALL features
    print("Building ECDF and transforming all 16 features to quantiles...")
    q_data = {}
    feature_names = [
        'delta_col', 'delta_labA', 'delta_labB', 'delta_hsv_sat', 
        'delta_tex', 'delta_gabor', 'delta_edge', 'delta_gradOri',
        'delta_hfreq', 'delta_entropy', 'delta_msLap2', 'delta_salSR',
        'delta_shape', 'delta_compactHull', 'delta_sizeRatio', 'delta_ctxDist'
    ]
    for feature in feature_names:
        ecdf = ECDF(data_df[feature].dropna())
        clipped_values = np.clip(data_df[feature], ecdf.x.min(), ecdf.x.max())
        q_data[f'q_{feature}'] = ecdf(clipped_values)
    
    q_df = pd.DataFrame(q_data)
    data_df = pd.concat([data_df, q_df], axis=1)

    # Create a preliminary c_raw from all features to determine level mapping direction
    all_q_cols = [f'q_{f}' for f in feature_names]
    data_df['c_raw_all'] = data_df[all_q_cols].mean(axis=1)

    # Step 2+.3: Automatic Feature Selection
    selected_features, selected_feature_names = select_features_forward_greedy(data_df, feature_names)
    
    # Drop the preliminary c_raw column
    data_df = data_df.drop(columns=['c_raw_all'])

    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "selected_features.json", "w") as f:
        json.dump({"selected_indices": selected_features, "selected_names": selected_feature_names}, f, indent=4)
    print(f"Selected {len(selected_feature_names)} features: {selected_feature_names}")

    # Step 4 (modified): Fuse quantiles of SELECTED features
    quantile_cols = [f'q_{f}' for f in selected_feature_names]
    data_df['c_raw'] = data_df[quantile_cols].mean(axis=1)

    # Step 5.1: Dual mapping self-check (using the new c_raw)
    print("Performing dual-mapping self-check...")
    cod_df = data_df[data_df['type'] == 'COD'].copy()
    
    cod_map_m1 = {255:1, 204:2, 153:3, 102:4, 51:5}
    cod_map_m2 = {255:5, 204:4, 153:3, 102:2, 51:1}
    
    # Calculate correlation for both mappings
    rho_m1_res = spearmanr(cod_df['c_raw'], pd.Series(cod_df['v']).map(cod_map_m1))
    rho_m2_res = spearmanr(cod_df['c_raw'], pd.Series(cod_df['v']).map(cod_map_m2))

    # Choose mapping with higher absolute correlation
    if abs(rho_m1_res.correlation) >= abs(rho_m2_res.correlation): # type: ignore
        print(f"Selected mapping M1 (ρ={rho_m1_res.correlation:.4f} vs M2 ρ={rho_m2_res.correlation:.4f})") # type: ignore
        cod_level_map = cod_map_m1
        sod_level_map = {153:6, 178:7, 204:8, 229:9, 255:10} # Original
    else:
        print(f"Selected mapping M2 (ρ={rho_m2_res.correlation:.4f} vs M1 ρ={rho_m1_res.correlation:.4f})") # type: ignore
        cod_level_map = cod_map_m2
        sod_level_map = {153:10, 178:9, 204:8, 229:7, 255:6} # Reversed
    
    # Apply final level mapping
    data_df['level'] = -1
    cod_mask_map = data_df['type'] == 'COD'
    sod_mask_map = data_df['type'] == 'SOD'
    data_df.loc[cod_mask_map, 'level'] = data_df.loc[cod_mask_map, 'v'].map(cod_level_map)
    data_df.loc[sod_mask_map, 'level'] = data_df.loc[sod_mask_map, 'v'].map(sod_level_map)

    # Step 5.3: Supervised Monotonic Calibration
    print("Applying supervised monotonic calibration...")
    cod_df = data_df[data_df['type'] == 'COD'].copy()
    sod_df = data_df[data_df['type'] == 'SOD'].copy()

    # Create target_y based on the chosen levels
    cod_df['target_y'] = cod_df['level'] / 5.0
    sod_df['target_y'] = (sod_df['level'] - 5.0) / 5.0

    # Calibrate COD
    iso_cod = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso_cod.fit(cod_df['c_raw'], cod_df['target_y'])
    data_df.loc[data_df['type'] == 'COD', 'c_star'] = iso_cod.transform(data_df.loc[data_df['type'] == 'COD', 'c_raw'])
    joblib.dump(iso_cod, isotonic_cod_path)
    print(f"Saved COD isotonic model to {isotonic_cod_path}")

    # Calibrate SOD
    iso_sod = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso_sod.fit(sod_df['c_raw'], sod_df['target_y'])
    data_df.loc[data_df['type'] == 'SOD', 'c_star'] = iso_sod.transform(data_df.loc[data_df['type'] == 'SOD', 'c_raw'])
    joblib.dump(iso_sod, isotonic_sod_path)
    print(f"Saved SOD isotonic model to {isotonic_sod_path}")

    # Step 6: Map to final score
    print("Mapping to final scores...")
    data_df['score'] = 0.0
    cod_mask = data_df['type'] == 'COD'
    sod_mask = data_df['type'] == 'SOD'
    
    data_df.loc[cod_mask, 'score'] = 5 * data_df.loc[cod_mask, 'c_star']
    data_df.loc[sod_mask, 'score'] = 5 + 5 * data_df.loc[sod_mask, 'c_star']

    # Spearman correlation calculation and final JSON logging
    metrics = calculate_spearman_metrics(data_df)
    print(f"Spearman ρ (COD | final) : {metrics['rho_cod_final']:.4f}")
    print(f"Spearman ρ (SOD | final) : {metrics['rho_sod_final']:.4f}")
    print(f"Spearman ρ (COD | vs raw) : {metrics['rho_cod_vs_raw']:.4f}")
    print(f"Spearman ρ (SOD | vs raw) : {metrics['rho_sod_vs_raw']:.4f}")

    # Step 2+.4: Check final correlation
    if metrics['rho_cod_final'] < 0.50:
        print(f"\n--- WARNING: FEATURE_INSUFFICIENT ---")
        print(f"Final COD Spearman correlation ({metrics['rho_cod_final']:.4f}) is below the 0.50 threshold.")
        print("Exiting.")
        return # Exit the script

    with open(output_dir / "spearman_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved Spearman metrics to {output_dir / 'spearman_metrics.json'}")

    # Generate diagnostic plots
    create_diagnostic_plots(data_df, metrics, output_dir)
    
    # Final Output
    print("Generating final CSV output...")
    output_df = data_df[['img_path', 'mask_path', 'score']].copy()
    output_df['img_path'] = output_df['img_path'].astype(str)
    output_df['mask_path'] = output_df['mask_path'].astype(str)
    output_df['score'] = output_df['score'].round(4)
    
    output_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Successfully generated {csv_path}")
    print("--- Pipeline Finished ---")


def get_file_list(dataset_root: Path):
    """
    Enumerates all mask files, determines their type (COD/SOD), validates them,
    and constructs the path to the corresponding image file.
    """
    mask_dirs = [
        dataset_root / "TrainDataset" / "Mask",
        dataset_root / "TestDataset" / "COD10K" / "Mask",
    ]

    cod_v_values = {51, 102, 153, 204, 255}
    sod_v_values = {153, 178, 204, 229, 255}

    all_files = []
    
    print("Enumerating mask files...")
    mask_paths = []
    for d in mask_dirs:
        mask_paths.extend(glob.glob(str(d / "*.png")))

    for mask_path_str in tqdm(mask_paths, desc="Validating files"):
        mask_path = Path(mask_path_str)
        fname = mask_path.stem
        
        parts = fname.rsplit('_', 1)
        if len(parts) != 2 or not parts[1].isdigit():
            warnings.warn(f"Skipping mask with unexpected name format: {mask_path.name}")
            continue
        
        name, v_str = parts
        v = int(v_str)

        file_type = None
        if name.startswith("COD10K") or name.startswith("camourflage"):
            if v in cod_v_values:
                file_type = "COD"
            else:
                warnings.warn(f"Skipping COD mask with unexpected v={v}: {mask_path.name}")
                continue
        elif name.startswith("COCO"):
            if v in sod_v_values:
                file_type = "SOD"
            else:
                warnings.warn(f"Skipping SOD mask with unexpected v={v}: {mask_path.name}")
                continue
        else:
            warnings.warn(f"Skipping mask with unknown type: {mask_path.name}")
            continue

        img_path = mask_path.parent.parent / "Imgs_full" / f"{fname}.jpg"
        if not img_path.exists():
            warnings.warn(f"Image file not found for mask, skipping: {img_path}")
            continue

        all_files.append({
            "mask_path": mask_path,
            "img_path": img_path,
            "type": file_type,
            "v": v
        })
        
    return all_files


def compute_all_features(img_path: Path, mask_path: Path):
    """
    Computes all 16 features for a given image-mask pair.
    """
    try:
        # Load image and mask
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: raise FileNotFoundError(f"Could not read image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None: raise FileNotFoundError(f"Could not read mask: {mask_path}")

        # --- Adaptive BG and FG masks ---
        fg_mask, bg_mask = compute_adaptive_background(mask)
        if fg_mask is None or bg_mask is None:
            return None # Skip if masks are invalid

        # --- Pre-computations for features ---
        h, w = mask.shape
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # --- Feature Extraction ---
        features = {}
        
        # 1-4: Color features
        mu_fg_lab = np.mean(img_lab[fg_mask], axis=0)
        mu_bg_lab = np.mean(img_lab[bg_mask], axis=0)
        features['delta_col'] = np.linalg.norm(mu_fg_lab - mu_bg_lab) / 100.0
        features['delta_labA'] = np.abs(mu_fg_lab[1] - mu_bg_lab[1])
        features['delta_labB'] = np.abs(mu_fg_lab[2] - mu_bg_lab[2])
        features['delta_hsv_sat'] = np.mean(img_hsv[:,:,1][fg_mask]) - np.mean(img_hsv[:,:,1][bg_mask])
        
        # 5-12: Texture/Gradient/Frequency features
        lbp = local_binary_pattern(img_gray, LBP_N_POINTS, LBP_RADIUS, method='uniform')
        features['delta_tex'] = np.abs(np.std(lbp[fg_mask]) - np.std(lbp[bg_mask]))
        
        gabor_kernels = [cv2.getGaborKernel((21, 21), 6.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F) for theta in np.arange(0, np.pi, np.pi / 5)]
        gabor_energy = np.mean([cv2.filter2D(img_gray, cv2.CV_32F, k) for k in gabor_kernels], axis=0)
        features['delta_gabor'] = np.mean(gabor_energy[fg_mask]) - np.mean(gabor_energy[bg_mask])
        
        edges = canny(img_gray.astype(float) / 255.0, sigma=1.0)
        features['delta_edge'] = np.mean(edges[fg_mask]) - np.mean(edges[bg_mask])
        
        _, hog_image = hog(img_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
        features['delta_gradOri'] = np.mean(hog_image[fg_mask]) - np.mean(hog_image[bg_mask])

        features['delta_hfreq'] = compute_hfreq_feature(img_gray, fg_mask, bg_mask)
        features['delta_entropy'] = compute_entropy_feature(img_gray, fg_mask, bg_mask)
        
        lap_vars = []
        for sigma in [2, 4, 8]:
            lap = gaussian_laplace(img_gray.astype(float), sigma=sigma)
            lap_vars.append(np.abs(np.var(lap[fg_mask]) - np.var(lap[bg_mask])))
        features['delta_msLap2'] = np.mean(lap_vars)
        
        try:
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create() # type: ignore
            _, sal_map = saliency.computeSaliency(img_rgb)
            features['delta_salSR'] = np.mean(sal_map[fg_mask]) - np.mean(sal_map[bg_mask])
        except AttributeError:
            if not hasattr(compute_all_features, 'saliency_warning_printed'):
                warnings.warn("cv2.saliency module not found. Is opencv-contrib-python installed? Setting delta_salSR to 0.", UserWarning)
                compute_all_features.saliency_warning_printed = True # type: ignore
            features['delta_salSR'] = 0.0


        # 13-16: Shape/Context features
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(main_contour)
            perimeter = cv2.arcLength(main_contour, True)
            hull = cv2.convexHull(main_contour)
            hull_area = cv2.contourArea(hull)

            features['delta_shape'] = 1 - ((4 * np.pi * area) / (perimeter**2)) if perimeter > 0 else 0
            features['delta_compactHull'] = area / hull_area if hull_area > 0 else 0
        else:
            features['delta_shape'] = 1.0
            features['delta_compactHull'] = 0.0

        features['delta_sizeRatio'] = fg_mask.sum() / (h * w)
        img_center = np.array([h / 2, w / 2])
        fg_center = np.array(center_of_mass(fg_mask))
        features['delta_ctxDist'] = np.linalg.norm(fg_center - img_center) / np.linalg.norm(img_center)

        features.update({"img_path": img_path, "mask_path": mask_path})
        return features

    except Exception as e:
        warnings.warn(f"Error processing {mask_path.name}: {e}")
        return None

def select_features_forward_greedy(df: pd.DataFrame, all_feature_names: list) -> tuple:
    """
    Selects the best subset of features using forward greedy selection based on Spearman correlation.
    """
    print("\n--- Starting Automatic Feature Selection ---")
    
    selected_indices = []
    selected_names = []
    best_rho = -1.0
    
    # Determine level mapping once for all candidates
    cod_df = df[df['type'] == 'COD'].copy()
    cod_map_m1 = {255:1, 204:2, 153:3, 102:4, 51:5}
    cod_map_m2 = {255:5, 204:4, 153:3, 102:2, 51:1}
    # Use the preliminary 'c_raw_all' to decide on mapping direction
    rho_m1_res = spearmanr(cod_df['c_raw_all'], pd.Series(cod_df['v']).map(cod_map_m1))
    rho_m2_res = spearmanr(cod_df['c_raw_all'], pd.Series(cod_df['v']).map(cod_map_m2))
    cod_level_map = cod_map_m1 if abs(rho_m1_res.correlation) >= abs(rho_m2_res.correlation) else cod_map_m2 # type: ignore
    cod_df['level'] = pd.Series(cod_df['v']).map(cod_level_map)

    for i in range(len(all_feature_names)):
        remaining_indices = [idx for idx, name in enumerate(all_feature_names) if name not in selected_names]
        
        if not remaining_indices:
            break

        best_candidate_idx = -1
        best_candidate_rho = -1.0
        
        for idx in remaining_indices:
            candidate_name = all_feature_names[idx]
            current_selection_names = selected_names + [candidate_name]
            
            # Fuse quantiles for current candidate set
            q_cols = [f'q_{name}' for name in current_selection_names]
            c_raw_tmp = cod_df[q_cols].mean(axis=1)

            # Supervised Isotonic Calibration
            target_y = cod_df['level'] / 5.0
            iso_tmp = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            iso_tmp.fit(c_raw_tmp, target_y)
            score_tmp = 5 * iso_tmp.transform(c_raw_tmp)
            
            # Evaluate
            rho_res = spearmanr(score_tmp, cod_df['level'])
            rho = rho_res.correlation # type: ignore
            
            if rho > best_candidate_rho:
                best_candidate_rho = rho
                best_candidate_idx = idx

        if best_candidate_rho - best_rho < 0.02:
            print(f"Stopping selection. Improvement Δρ ({best_candidate_rho - best_rho:.4f}) is less than 0.02.")
            break
        
        best_rho = best_candidate_rho
        selected_indices.append(best_candidate_idx)
        selected_names.append(all_feature_names[best_candidate_idx])
        print(f"  > Selected feature {i+1}: '{all_feature_names[best_candidate_idx]}' (New ρ = {best_rho:.4f})")
    
    print("--- Finished Feature Selection ---\n")
    return selected_indices, selected_names

def compute_adaptive_background(mask: np.ndarray, min_ring_ratio=0.01, max_retries=3) -> tuple:
    """Computes FG and adaptive BG masks."""
    fg_mask = mask > 127
    if not fg_mask.any():
        warnings.warn("Mask is empty")
        return None, None

    total_pixels = mask.size
    fg_area = fg_mask.sum()
    
    w = int(np.clip(0.1 * np.sqrt(fg_area), 4, 32))

    for i in range(max_retries):
        kernel_size = w * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1) > 127
        bg_mask = dilated_mask & ~fg_mask
        
        if bg_mask.sum() / total_pixels > min_ring_ratio:
            return fg_mask, bg_mask
        
        w *= 2 # Double width and retry
    
    warnings.warn(f"Could not create sufficient background ring after {max_retries} retries.")
    return fg_mask, bg_mask # Return what we have


def compute_hfreq_feature(gray_img: np.ndarray, fg_mask: np.ndarray, bg_mask: np.ndarray) -> float:
    """Computes the high-frequency energy feature."""
    h, w = gray_img.shape
    
    # FFT
    f = fft2(gray_img)
    fshift = fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)

    # Frequency grid
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    radius = np.sqrt(x*x + y*y)
    
    # Nyquist frequency is at radius h/2 or w/2. We take the smaller one to be safe.
    nyquist = min(h, w) / 2
    
    # Band-pass mask for 0.25-0.5 Nyquist
    bp_mask = (radius > 0.25 * nyquist) & (radius <= 0.5 * nyquist)

    # Mean high-frequency energy for FG and BG
    fg_hfreq = np.mean(magnitude_spectrum[fg_mask & bp_mask]) if (fg_mask & bp_mask).any() else 0.0
    bg_hfreq = np.mean(magnitude_spectrum[bg_mask & bp_mask]) if (bg_mask & bp_mask).any() else 0.0

    return float(fg_hfreq - bg_hfreq)

def compute_entropy_feature(gray_img: np.ndarray, fg_mask: np.ndarray, bg_mask: np.ndarray) -> float:
    """Computes the Shannon entropy feature."""
    # skimage.measure.shannon_entropy converts to uint8 internally
    fg_entropy = shannon_entropy(gray_img[fg_mask]) if fg_mask.any() else 0.0
    bg_entropy = shannon_entropy(gray_img[bg_mask]) if bg_mask.any() else 0.0
    
    return float(fg_entropy - bg_entropy)

def calculate_spearman_metrics(df: pd.DataFrame) -> dict:
    """Calculates all spearman correlation metrics."""
    cod_df = df[df['type'] == 'COD'].copy()
    sod_df = df[df['type'] == 'SOD'].copy()
    
    # Metric 1: COD score vs level
    rho_cod_level_res = spearmanr(cod_df['score'], cod_df['level'])

    # Metric 2: SOD score vs level (averaged over images)
    sod_image_rhos = []
    # Create a temporary column for grouping to help the linter
    sod_df_copy = sod_df.copy()
    sod_df_copy['img_base_name'] = pd.Series(sod_df_copy['img_path']).apply(lambda p: Path(p).stem.rsplit('_', 1)[0])
    if 'img_base_name' in sod_df_copy.columns:
        sod_groups = sod_df_copy.groupby('img_base_name')
        for _, grp in sod_groups:
            if pd.Series(grp['level']).nunique() > 1:
                res = spearmanr(grp['score'], grp['level'])
                if not np.isnan(res.correlation): # type: ignore
                    sod_image_rhos.append(res.correlation) # type: ignore
    rho_sod_level = np.nanmean(sod_image_rhos) if sod_image_rhos else 0.0

    # Metric 3 & 4: score vs c_raw for COD and SOD
    rho_cod_raw_res = spearmanr(cod_df['score'], cod_df['c_raw'])
    rho_sod_raw_res = spearmanr(sod_df['score'], sod_df['c_raw'])

    return {
        "rho_cod_final": rho_cod_level_res.correlation, # type: ignore
        "rho_sod_final": rho_sod_level,
        "rho_cod_vs_raw": rho_cod_raw_res.correlation, # type: ignore
        "rho_sod_vs_raw": rho_sod_raw_res.correlation, # type: ignore
    }

def create_diagnostic_plots(df: pd.DataFrame, metrics: dict, output_dir: Path):
    """Generates and saves a figure with diagnostic plots."""
    print("Generating diagnostic plots...")
    # Import here to avoid making it a dependency for the whole script
    import matplotlib.pyplot as plt
    
    cod_df = df[df['type'] == 'COD']
    sod_df = df[df['type'] == 'SOD']

    # --- Figure Setup ---
    # Convert mm to inches for figsize, 220mm x 150mm
    fig = plt.figure(figsize=(220 / 25.4, 150 / 25.4), dpi=200)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    fig.suptitle("Relationship Between Raw Contrast, Discrete Level, and Final Score")

    # --- Subplot A: c_raw vs score ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hexbin(cod_df['c_raw'], cod_df['score'], gridsize=50, cmap='Blues', label='COD', mincnt=1)
    ax1.hexbin(sod_df['c_raw'], sod_df['score'], gridsize=50, cmap='Reds', label='SOD', mincnt=1, alpha=0.8)
    ax1.set_xlabel("Raw Contrast C_raw")
    ax1.set_ylabel("Mapped Score")
    ax1.set_title("A: Mapped Score vs. Raw Contrast")
    
    text_str = (
        f"$\\rho_{{COD}}$ (vs raw): {metrics['rho_cod_vs_raw']:.3f}\n"
        f"$\\rho_{{SOD}}$ (vs raw): {metrics['rho_sod_vs_raw']:.3f}"
    )
    ax1.text(0.05, 0.95, text_str, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))

    # --- Subplot B: Score Histogram ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(cod_df['score'], bins=25, range=(0, 5), density=True, alpha=0.7, label='COD')
    ax2.hist(sod_df['score'], bins=25, range=(5, 10), density=True, alpha=0.7, label='SOD')
    ax2.set_xlabel("Mapped Score")
    ax2.set_ylabel("Frequency")
    ax2.set_title("B: Score Distribution")
    ax2.legend()

    # --- Subplot C: Score vs Level Boxplot ---
    ax3 = fig.add_subplot(gs[1, :])
    levels = sorted(df['level'].unique())
    data_to_plot = [df[df['level'] == lvl]['score'] for lvl in levels]
    ax3.boxplot(data_to_plot, patch_artist=True,
                boxprops=dict(facecolor='lightblue'))
    ax3.set_xticklabels(labels=levels)
    
    # Ideal trend line
    ideal_scores = []
    for l in levels:
        if 1 <= l <= 5:
            ideal_scores.append(5 * (l - 1) / 4.0)
        else:
            ideal_scores.append(5 + 5 * (l - 6) / 4.0)
    ax3.plot(range(1, len(levels) + 1), ideal_scores, 'r--', label='Ideal Trend')

    ax3.set_xlabel("Discrete Level")
    ax3.set_ylabel("Mapped Score")
    ax3.set_title("C: Final Score vs. Discrete Level")
    ax3.legend()

    # --- Save Figure ---
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    png_path = output_dir / "score_distribution.png"
    pdf_path = output_dir / "score_distribution.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    print(f"Saved diagnostic plots to {png_path} and {pdf_path}")


if __name__ == '__main__':
    main() 